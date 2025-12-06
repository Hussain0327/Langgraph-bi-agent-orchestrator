import json
from typing import Dict, Any, List, Optional, Tuple
from pydantic import ValidationError
from .agent_output import AgentOutput, SynthesisOutput, Metric, DataTable, ChartSpec, Recommendation, Citation, Findings

class ValidationResult:

    def __init__(self, valid: bool, errors: List[str]=None, warnings: List[str]=None):
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []

    def __bool__(self):
        return self.valid

    def __str__(self):
        if self.valid:
            msg = '✓ Validation passed'
            if self.warnings:
                msg += f'\n  {len(self.warnings)} warnings:\n'
                msg += '\n'.join((f'  - {w}' for w in self.warnings))
            return msg
        else:
            msg = f'✗ Validation failed with {len(self.errors)} errors:\n'
            msg += '\n'.join((f'  - {e}' for e in self.errors))
            if self.warnings:
                msg += f'\n  {len(self.warnings)} warnings:\n'
                msg += '\n'.join((f'  - {w}' for w in self.warnings))
            return msg

def validate_agent_output(data: Dict[str, Any]) -> ValidationResult:
    errors = []
    warnings = []
    try:
        agent_output = AgentOutput(**data)
        quality_result = check_output_quality(agent_output)
        warnings.extend(quality_result.warnings)
        return ValidationResult(valid=True, warnings=warnings)
    except ValidationError as e:
        for error in e.errors():
            field = ' -> '.join((str(x) for x in error['loc']))
            message = error['msg']
            errors.append(f'{field}: {message}')
        return ValidationResult(valid=False, errors=errors)
    except Exception as e:
        errors.append(f'Unexpected error: {str(e)}')
        return ValidationResult(valid=False, errors=errors)

def validate_synthesis_output(data: Dict[str, Any]) -> ValidationResult:
    errors = []
    warnings = []
    try:
        synthesis_output = SynthesisOutput(**data)
        quality_result = check_synthesis_quality(synthesis_output)
        warnings.extend(quality_result.warnings)
        return ValidationResult(valid=True, warnings=warnings)
    except ValidationError as e:
        for error in e.errors():
            field = ' -> '.join((str(x) for x in error['loc']))
            message = error['msg']
            errors.append(f'{field}: {message}')
        return ValidationResult(valid=False, errors=errors)
    except Exception as e:
        errors.append(f'Unexpected error: {str(e)}')
        return ValidationResult(valid=False, errors=errors)

def check_output_quality(output: AgentOutput) -> ValidationResult:
    warnings = []
    placeholder_check = check_for_placeholders(output.model_dump())
    if placeholder_check:
        warnings.extend(placeholder_check)
    if len(output.findings.executive_summary) < 50:
        warnings.append('Executive summary is very short (< 50 chars)')
    elif len(output.findings.executive_summary) > 1000:
        warnings.append('Executive summary is very long (> 1000 chars)')
    if len(output.findings.key_findings) < 3:
        warnings.append('Fewer than 3 key findings (recommended: 3-5)')
    elif len(output.findings.key_findings) > 7:
        warnings.append('More than 7 key findings (recommended: 3-5, consider consolidating)')
    if not output.findings.metrics:
        warnings.append('No metrics provided - consider adding quantitative data')
    if not output.findings.recommendations:
        warnings.append('No recommendations provided')
    for i, chart in enumerate(output.findings.chart_specs):
        chart_warnings = validate_chart_spec(chart, f'Chart {i + 1}')
        warnings.extend(chart_warnings)
    for i, table in enumerate(output.findings.data_tables):
        table_warnings = validate_data_table(table, f'Table {i + 1}')
        warnings.extend(table_warnings)
    return ValidationResult(valid=True, warnings=warnings)

def check_synthesis_quality(synthesis: SynthesisOutput) -> ValidationResult:
    warnings = []
    if not synthesis.agent_outputs:
        warnings.append('No individual agent outputs included in synthesis')
    consulted_set = set(synthesis.agents_consulted)
    outputs_set = set(synthesis.agent_outputs.keys())
    if consulted_set != outputs_set:
        missing = consulted_set - outputs_set
        extra = outputs_set - consulted_set
        if missing:
            warnings.append(f'Agents consulted but no output: {missing}')
        if extra:
            warnings.append(f'Agent outputs provided but not in consulted list: {extra}')
    for agent_name, agent_output in synthesis.agent_outputs.items():
        quality_result = check_output_quality(agent_output)
        for warning in quality_result.warnings:
            warnings.append(f'{agent_name}: {warning}')
    synthesis_quality = check_output_quality(AgentOutput(query=synthesis.query, agent='market', findings=synthesis.synthesis, metadata={'confidence': synthesis.overall_confidence, 'model': 'synthesis'}))
    warnings.extend(synthesis_quality.warnings)
    return ValidationResult(valid=True, warnings=warnings)

def check_for_placeholders(data: Dict[str, Any], path: str='') -> List[str]:
    placeholders = ['TBD', 'TODO', 'FIXME', 'XXX', 'PLACEHOLDER', 'FILL_IN', 'CHANGE_ME']
    warnings = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f'{path}.{key}' if path else key
            warnings.extend(check_for_placeholders(value, new_path))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_path = f'{path}[{i}]'
            warnings.extend(check_for_placeholders(item, new_path))
    elif isinstance(data, str):
        for placeholder in placeholders:
            if placeholder in data.upper():
                warnings.append(f"Placeholder '{placeholder}' found in {path}")
    return warnings

def validate_chart_spec(chart: ChartSpec, name: str='Chart') -> List[str]:
    warnings = []
    if len(chart.x_data) == 0:
        warnings.append(f'{name}: No data points')
    elif len(chart.x_data) == 1:
        warnings.append(f'{name}: Only 1 data point (charts need 2+)')
    if len(chart.x_data) > 50:
        warnings.append(f'{name}: {len(chart.x_data)} data points (may be hard to read, consider aggregating)')
    if not chart.title:
        warnings.append(f'{name}: Missing title')
    elif len(chart.title) > 80:
        warnings.append(f'{name}: Title very long (> 80 chars)')
    if chart.colors:
        if len(chart.colors) != len(chart.x_data):
            warnings.append(f"{name}: Number of colors ({len(chart.colors)}) doesn't match data points ({len(chart.x_data)})")
        for color in chart.colors:
            if not color.startswith('#') or len(color) != 7:
                warnings.append(f"{name}: Invalid color format '{color}' (should be #RRGGBB)")
    return warnings

def validate_data_table(table: DataTable, name: str='Table') -> List[str]:
    warnings = []
    if len(table.rows) == 0:
        warnings.append(f'{name}: No data rows')
    if len(table.rows) > 1000:
        warnings.append(f'{name}: {len(table.rows)} rows (consider pagination or summary)')
    for col in table.columns:
        if not col:
            warnings.append(f'{name}: Empty column name')
        if len(col) > 50:
            warnings.append(f"{name}: Column name very long: '{col}'")
    if len(table.columns) != len(set(table.columns)):
        duplicates = [col for col in table.columns if table.columns.count(col) > 1]
        warnings.append(f'{name}: Duplicate column names: {set(duplicates)}')
    return warnings

def validate_metrics_consistency(output: AgentOutput) -> ValidationResult:
    warnings = []
    metric_names = set(output.findings.metrics.keys())
    narrative_text = output.findings.executive_summary + ' ' + output.findings.narrative + ' ' + ' '.join(output.findings.key_findings)
    mentioned_metrics = set()
    for metric_name in metric_names:
        variations = [metric_name, metric_name.upper(), metric_name.lower(), metric_name.replace('_', ' ')]
        if any((var in narrative_text for var in variations)):
            mentioned_metrics.add(metric_name)
    unmentioned = metric_names - mentioned_metrics
    if unmentioned:
        warnings.append(f'Metrics defined but not mentioned in text: {unmentioned}')
    return ValidationResult(valid=True, warnings=warnings)

def validate_json_file(file_path: str) -> ValidationResult:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return validate_agent_output(data)
    except FileNotFoundError:
        return ValidationResult(valid=False, errors=[f'File not found: {file_path}'])
    except json.JSONDecodeError as e:
        return ValidationResult(valid=False, errors=[f'Invalid JSON: {str(e)}'])
    except Exception as e:
        return ValidationResult(valid=False, errors=[f'Error reading file: {str(e)}'])

def export_json_schema(output_path: str='schemas/agent_output_schema.json'):
    schema = AgentOutput.model_json_schema()
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)
    print(f'✓ JSON schema exported to: {output_path}')
if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Validate agent output JSON')
    parser.add_argument('file', help='JSON file to validate')
    parser.add_argument('--export-schema', help='Export JSON schema to file')
    args = parser.parse_args()
    if args.export_schema:
        export_json_schema(args.export_schema)
        sys.exit(0)
    result = validate_json_file(args.file)
    print(result)
    sys.exit(0 if result.valid else 1)