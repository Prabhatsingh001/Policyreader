#!/usr/bin/env python3
"""
Main entry point for the Intelligent Query-Retrieval System.
This script provides a simple interface to the RAG system.
"""

import argparse
import sys
from pathlib import Path
from rag_system import IntelligentQuerySystem
import os

def main():
    """Main function to run the RAG system."""
    parser = argparse.ArgumentParser(
        description="Intelligent Query-Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                           # Run the demo
  python main.py --query "What is the coverage limit?"  # Single query
  python main.py --docs ./documents --query "..." # Process docs and query
  python main.py --api                            # Start API server
        """
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run the demonstration with sample documents"
    )
    
    parser.add_argument(
        "--docs", 
        type=str,
        help="Path to directory containing documents to process"
    )
    
    parser.add_argument(
        "--query", 
        type=str,
        help="Query to process"
    )
    
    parser.add_argument(
        "--api", 
        action="store_true",
        help="Start the API server"
    )
    
    parser.add_argument(
        "--port", 
        type=int,
        default=8000,
        help="Port for API server (default: 8000)"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run system tests"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Run tests
    if args.test:
        print("Running system tests...")
        from test_system import main as run_tests
        run_tests()
        return
    
    # Start API server
    if args.api or os.environ.get("RENDER") or os.environ.get("PORT"):
        port = int(os.environ.get("PORT", args.port))
        print(f"Starting API server on port {port}...")
        from api_server import app
        import uvicorn
        uvicorn.run("api_server:app", host="0.0.0.0", port=port)
        return
    
    # Run demo
    if args.demo:
        print("Running demo...")
        from demo import run_demo
        run_demo()
        return
    
    # Initialize the RAG system
    print("Initializing Intelligent Query-Retrieval System...")
    system = IntelligentQuerySystem()
    
    # Process documents if provided
    if args.docs:
        docs_path = Path(args.docs)
        if not docs_path.exists():
            print(f"Error: Directory {args.docs} does not exist")
            return
        
        print(f"Processing documents from: {args.docs}")
        system.add_documents(str(docs_path))
        
        # Show system stats
        stats = system.get_system_stats()
        print(f"System loaded with {stats['vector_store']['total_chunks']} document chunks")
    
    # Process query if provided
    if args.query:
        print(f"\nProcessing query: {args.query}")
        print("-" * 50)
        
        try:
            response = system.query(args.query)
            
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Domain: {response.domain}")
            print(f"Sources: {', '.join(response.sources)}")
            print(f"Relevant Clauses: {', '.join(response.relevant_clauses)}")
            print(f"Reasoning: {response.reasoning}")
            
        except Exception as e:
            print(f"Error processing query: {e}")
    
    # If no query provided but docs were processed, start interactive mode
    if args.docs and not args.query:
        print("\nEntering interactive mode. Type 'quit' to exit.")
        print("Enter your queries:")
        
        while True:
            try:
                query = input("\n> ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if query:
                    response = system.query(query)
                    print(f"\nAnswer: {response.answer}")
                    print(f"Confidence: {response.confidence:.2f}")
                    print(f"Domain: {response.domain}")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
