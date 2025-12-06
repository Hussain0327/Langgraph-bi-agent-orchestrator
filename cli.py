import os
import sys
from dotenv import load_dotenv
from colorama import init, Fore, Style
sys.path.insert(0, os.path.dirname(__file__))
from src.langgraph_orchestrator import LangGraphOrchestrator
from src.config import Config
init(autoreset=True)

def print_banner():
    gpt5_status = '✓ GPT-5' if Config.is_gpt5() else f'GPT-4'
    langsmith_status = '✓ LangSmith ON' if Config.LANGCHAIN_TRACING_V2 else 'LangSmith OFF'
    banner = f"\n{Fore.CYAN}{'=' * 70}\n{Fore.CYAN}  Business Intelligence Orchestrator v2 - CLI\n{Fore.CYAN}  LangGraph + GPT-5 + LangSmith Multi-Agent System\n{Fore.GREEN}  {gpt5_status} | {langsmith_status}\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n"
    print(banner)

def print_agents(agents):
    if agents:
        print(f"\n{Fore.YELLOW}→ Consulting Agents: {', '.join(agents)}{Style.RESET_ALL}")

def print_detailed_findings(findings):
    print(f"\n{Fore.CYAN}{'=' * 70}")
    print(f'DETAILED AGENT FINDINGS')
    print(f"{'=' * 70}{Style.RESET_ALL}\n")
    if 'market_analysis' in findings:
        print(f'{Fore.GREEN} MARKET ANALYSIS:{Style.RESET_ALL}')
        print(findings['market_analysis'])
        print()
    if 'operations_audit' in findings:
        print(f'{Fore.GREEN}⚙  OPERATIONS AUDIT:{Style.RESET_ALL}')
        print(findings['operations_audit'])
        print()
    if 'financial_modeling' in findings:
        print(f'{Fore.GREEN} FINANCIAL MODELING:{Style.RESET_ALL}')
        print(findings['financial_modeling'])
        print()
    if 'lead_generation' in findings:
        print(f'{Fore.GREEN} LEAD GENERATION:{Style.RESET_ALL}')
        print(findings['lead_generation'])
        print()

def print_synthesis(synthesis):
    print(f"{Fore.CYAN}{'=' * 70}")
    print(f'SYNTHESIZED RECOMMENDATION')
    print(f"{'=' * 70}{Style.RESET_ALL}\n")
    print(f'{Fore.WHITE}{synthesis}{Style.RESET_ALL}\n')

def main():
    load_dotenv()
    try:
        Config.validate()
    except ValueError as e:
        print(f'{Fore.RED}Error: {str(e)}{Style.RESET_ALL}')
        print(f'Please create a .env file with your OpenAI API key.')
        print(f'You can copy .env.example and fill in your key.')
        sys.exit(1)
    print_banner()
    print(f'{Fore.WHITE}✨ New Features:{Style.RESET_ALL}')
    print(f'  • {Fore.GREEN}LangGraph State Machine{Style.RESET_ALL}: Intelligent agent routing')
    print(f'  • {Fore.GREEN}GPT-5 Responses API{Style.RESET_ALL}: 40-80% cost reduction via caching')
    print(f'  • {Fore.GREEN}Semantic Routing{Style.RESET_ALL}: AI-powered (not keyword matching)')
    if Config.LANGCHAIN_TRACING_V2:
        print(f'  • {Fore.GREEN}LangSmith Tracing{Style.RESET_ALL}: Full observability in {Config.LANGCHAIN_PROJECT}')
    print()
    print(f'{Fore.WHITE}Available specialized agents:{Style.RESET_ALL}')
    print(f'  • {Fore.GREEN}Market Analysis{Style.RESET_ALL}: Market research, trends, competition')
    print(f'  • {Fore.GREEN}Operations Audit{Style.RESET_ALL}: Process optimization, efficiency')
    print(f'  • {Fore.GREEN}Financial Modeling{Style.RESET_ALL}: ROI, projections, cost analysis')
    print(f'  • {Fore.GREEN}Lead Generation{Style.RESET_ALL}: Customer acquisition, growth strategies')
    print()
    print(f'{Fore.YELLOW}Initializing LangGraph orchestrator...{Style.RESET_ALL}')
    try:
        orchestrator = LangGraphOrchestrator()
        print(f'{Fore.GREEN}✓ LangGraph orchestrator ready!{Style.RESET_ALL}')
        print(f'{Fore.GREEN}✓ Model: {Config.OPENAI_MODEL}{Style.RESET_ALL}\n')
    except Exception as e:
        print(f'{Fore.RED}Error initializing orchestrator: {str(e)}{Style.RESET_ALL}')
        sys.exit(1)
    print(f'{Fore.WHITE}Commands:{Style.RESET_ALL}')
    print(f'  • Type your business query and press Enter')
    print(f"  • Type 'history' to see conversation history")
    print(f"  • Type 'clear' to clear conversation memory")
    print(f"  • Type 'quit' or 'exit' to exit")
    print()
    while True:
        try:
            user_input = input(f'{Fore.CYAN}Query> {Style.RESET_ALL}').strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(f'\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}')
                break
            elif user_input.lower() == 'clear':
                orchestrator.clear_memory()
                print(f'{Fore.GREEN}✓ Conversation memory cleared{Style.RESET_ALL}\n')
                continue
            elif user_input.lower() == 'history':
                history = orchestrator.get_conversation_history()
                if history:
                    print(f'\n{Fore.CYAN}Conversation History:{Style.RESET_ALL}')
                    for msg in history:
                        role_color = Fore.GREEN if msg['role'] == 'user' else Fore.YELLOW
                        print(f"{role_color}{msg['role'].upper()}:{Style.RESET_ALL} {msg['content'][:100]}...")
                    print()
                else:
                    print(f'{Fore.YELLOW}No conversation history yet{Style.RESET_ALL}\n')
                continue
            print(f'\n{Fore.YELLOW}🔄 Analyzing query...{Style.RESET_ALL}')
            result = orchestrator.orchestrate(user_input)
            print_agents(result['agents_consulted'])
            print_detailed_findings(result['detailed_findings'])
            print_synthesis(result['recommendation'])
        except KeyboardInterrupt:
            print(f'\n\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}')
            break
        except Exception as e:
            print(f'\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}\n')
if __name__ == '__main__':
    main()