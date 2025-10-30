#!/usr/bin/env python3
"""CLI interface for the Business Intelligence Orchestrator."""
import os
import sys
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.orchestrator import PrimaryOrchestrator

# Initialize colorama for colored output
init(autoreset=True)


def print_banner():
    """Print the application banner."""
    banner = f"""
{Fore.CYAN}{'='*70}
{Fore.CYAN}  Business Intelligence Orchestrator - CLI
{Fore.CYAN}  Multi-Agent System for Comprehensive Business Analysis
{Fore.CYAN}{'='*70}{Style.RESET_ALL}
"""
    print(banner)


def print_agents(agents):
    """Print which agents will be consulted."""
    if agents:
        print(f"\n{Fore.YELLOW}→ Consulting Agents: {', '.join(agents)}{Style.RESET_ALL}")


def print_detailed_findings(findings):
    """Print detailed findings from each agent."""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"DETAILED AGENT FINDINGS")
    print(f"{'='*70}{Style.RESET_ALL}\n")

    if "market_analysis" in findings:
        print(f"{Fore.GREEN}📊 MARKET ANALYSIS:{Style.RESET_ALL}")
        print(findings["market_analysis"])
        print()

    if "operations_audit" in findings:
        print(f"{Fore.GREEN}⚙️  OPERATIONS AUDIT:{Style.RESET_ALL}")
        print(findings["operations_audit"])
        print()

    if "financial_modeling" in findings:
        print(f"{Fore.GREEN}💰 FINANCIAL MODELING:{Style.RESET_ALL}")
        print(findings["financial_modeling"])
        print()

    if "lead_generation" in findings:
        print(f"{Fore.GREEN}🎯 LEAD GENERATION:{Style.RESET_ALL}")
        print(findings["lead_generation"])
        print()


def print_synthesis(synthesis):
    """Print the synthesized recommendation."""
    print(f"{Fore.CYAN}{'='*70}")
    print(f"SYNTHESIZED RECOMMENDATION")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    print(f"{Fore.WHITE}{synthesis}{Style.RESET_ALL}\n")


def main():
    """Main CLI loop."""
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{Fore.RED}Error: OPENAI_API_KEY not found in environment variables.{Style.RESET_ALL}")
        print(f"Please create a .env file with your OpenAI API key.")
        print(f"You can copy .env.example and fill in your key.")
        sys.exit(1)

    print_banner()

    print(f"{Fore.WHITE}Available specialized agents:{Style.RESET_ALL}")
    print(f"  • {Fore.GREEN}Market Analysis{Style.RESET_ALL}: Market research, trends, competition")
    print(f"  • {Fore.GREEN}Operations Audit{Style.RESET_ALL}: Process optimization, efficiency")
    print(f"  • {Fore.GREEN}Financial Modeling{Style.RESET_ALL}: ROI, projections, cost analysis")
    print(f"  • {Fore.GREEN}Lead Generation{Style.RESET_ALL}: Customer acquisition, growth strategies")
    print()

    # Initialize orchestrator
    print(f"{Fore.YELLOW}Initializing orchestrator...{Style.RESET_ALL}")
    try:
        orchestrator = PrimaryOrchestrator()
        print(f"{Fore.GREEN}✓ Orchestrator ready!{Style.RESET_ALL}\n")
    except Exception as e:
        print(f"{Fore.RED}Error initializing orchestrator: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)

    print(f"{Fore.WHITE}Commands:{Style.RESET_ALL}")
    print(f"  • Type your business query and press Enter")
    print(f"  • Type 'history' to see conversation history")
    print(f"  • Type 'clear' to clear conversation memory")
    print(f"  • Type 'quit' or 'exit' to exit")
    print()

    # Main loop
    while True:
        try:
            # Get user input
            user_input = input(f"{Fore.CYAN}Query> {Style.RESET_ALL}").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break

            elif user_input.lower() == 'clear':
                orchestrator.clear_memory()
                print(f"{Fore.GREEN}✓ Conversation memory cleared{Style.RESET_ALL}\n")
                continue

            elif user_input.lower() == 'history':
                history = orchestrator.get_conversation_history()
                if history:
                    print(f"\n{Fore.CYAN}Conversation History:{Style.RESET_ALL}")
                    for msg in history:
                        role_color = Fore.GREEN if msg['role'] == 'user' else Fore.YELLOW
                        print(f"{role_color}{msg['role'].upper()}:{Style.RESET_ALL} {msg['content'][:100]}...")
                    print()
                else:
                    print(f"{Fore.YELLOW}No conversation history yet{Style.RESET_ALL}\n")
                continue

            # Process business query
            print(f"\n{Fore.YELLOW}🔄 Analyzing query...{Style.RESET_ALL}")

            result = orchestrator.orchestrate(user_input)

            # Print results
            print_agents(result["agents_consulted"])
            print_detailed_findings(result["detailed_findings"])
            print_synthesis(result["recommendation"])

        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
