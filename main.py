#!/usr/bin/env python3
"""
VectorCore - High-Performance Vector Database

A specialized in-memory database for storing and searching through millions of 
AI-generated vectors with extreme speed using k-d tree indexing.

Usage:
    python main.py [--host HOST] [--port PORT]

Commands (via network client):
    ADD doc_id [vector]     - Store a vector with document ID
    QUERY k [vector]        - Find k most similar vectors
    STATS                   - Get database statistics
    SAVE [filename]         - Save database to disk
    LOAD [filename]         - Load database from disk
    HELP                    - Show available commands

Example:
    ADD doc_123 [0.1, 0.2, 0.3]
    QUERY 5 [0.1, 0.2, 0.3]
"""

import argparse
import sys
from server import VectorCoreServer


def main():
    """Main entry point for VectorCore server."""
    parser = argparse.ArgumentParser(
        description="VectorCore - High-Performance Vector Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--host', 
        default='localhost',
        help='Host to bind the server to (default: localhost)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8888,
        help='Port to bind the server to (default: 8888)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='VectorCore 1.0.0'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 VectorCore - High-Performance Vector Database")
    print("=" * 60)
    print(f"🏠 Host: {args.host}")
    print(f"🚪 Port: {args.port}")
    print("=" * 60)
    
    # Create and start the server
    server = VectorCoreServer(host=args.host, port=args.port)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal...")
        server.stop()
        print("👋 VectorCore server stopped gracefully")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to start VectorCore server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 