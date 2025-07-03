import json
import socket
import threading
import time
from typing import Dict, List, Any
from vectorcore import VectorCore


class VectorCoreServer:
    """
    Network server for VectorCore database.
    
    Handles TCP connections and parses commands:
    - ADD doc_id [vector]
    - QUERY k [vector]
    - STATS
    - SAVE [filename]
    - LOAD [filename]
    - HELP
    """
    
    def __init__(self, host: str = 'localhost', port: int = 8888):
        self.host = host
        self.port = port
        self.vectorcore = VectorCore()
        self.server_socket = None
        self.running = False
        self.client_count = 0
        
    def start(self):
        """Start the VectorCore server."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.running = True
            
            print(f"🚀 VectorCore Server started on {self.host}:{self.port}")
            print("📡 Ready to accept connections...")
            print("💡 Use 'HELP' command for available operations")
            print("-" * 50)
            
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    self.client_count += 1
                    print(f"🔗 Client connected from {address} (Total clients: {self.client_count})")
                    
                    # Handle each client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        print(f"❌ Socket error: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the VectorCore server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("🛑 VectorCore Server stopped")
    
    def handle_client(self, client_socket: socket.socket, address):
        """Handle commands from a connected client."""
        try:
            # Send welcome message
            welcome_msg = (
                "🎯 Welcome to VectorCore Database!\n"
                "Type 'HELP' for available commands.\n"
                "Ready for commands...\n"
            )
            self.send_response(client_socket, welcome_msg)
            
            while self.running:
                try:
                    # Receive command from client
                    data = client_socket.recv(4096).decode('utf-8').strip()
                    if not data:
                        break
                    
                    print(f"📥 Command from {address}: {data[:100]}...")  # Truncate long commands in log
                    
                    # Process the command
                    start_time = time.time()
                    response = self.process_command(data)
                    processing_time = time.time() - start_time
                    
                    # Add timing information
                    response += f"\n⏱️  Processing time: {processing_time:.4f}s"
                    
                    # Send response back to client
                    self.send_response(client_socket, response)
                    
                except socket.timeout:
                    continue
                except socket.error as e:
                    print(f"⚠️  Client {address} connection error: {e}")
                    break
                except Exception as e:
                    error_response = f"❌ Error processing command: {str(e)}"
                    self.send_response(client_socket, error_response)
                    
        except Exception as e:
            print(f"❌ Error handling client {address}: {e}")
        finally:
            client_socket.close()
            self.client_count -= 1
            print(f"👋 Client {address} disconnected (Remaining clients: {self.client_count})")
    
    def send_response(self, client_socket: socket.socket, response: str):
        """Send response to client with proper formatting."""
        try:
            # Add newline for better formatting
            formatted_response = response + "\n>>> "
            client_socket.send(formatted_response.encode('utf-8'))
        except socket.error as e:
            print(f"⚠️  Error sending response: {e}")
    
    def process_command(self, command: str) -> str:
        """Process a command and return the response."""
        try:
            parts = command.strip().split()
            if not parts:
                return "❌ Empty command"
            
            cmd = parts[0].upper()
            
            if cmd == "HELP":
                return self.get_help_text()
            
            elif cmd == "ADD":
                return self.handle_add_command(parts[1:])
            
            elif cmd == "QUERY":
                return self.handle_query_command(parts[1:])
            
            elif cmd == "STATS":
                return self.handle_stats_command()
            
            elif cmd == "SAVE":
                filename = parts[1] if len(parts) > 1 else "vectorcore_data.pkl"
                return self.handle_save_command(filename)
            
            elif cmd == "LOAD":
                filename = parts[1] if len(parts) > 1 else "vectorcore_data.pkl"
                return self.handle_load_command(filename)
            
            elif cmd == "CLEAR":
                return self.handle_clear_command()
            
            elif cmd == "GET":
                if len(parts) < 2:
                    return "❌ Usage: GET doc_id"
                return self.handle_get_command(parts[1])
            
            elif cmd == "REMOVE":
                if len(parts) < 2:
                    return "❌ Usage: REMOVE doc_id"
                return self.handle_remove_command(parts[1])
            
            else:
                return f"❌ Unknown command: {cmd}. Type 'HELP' for available commands."
                
        except Exception as e:
            return f"❌ Error processing command: {str(e)}"
    
    def handle_add_command(self, args: List[str]) -> str:
        """Handle ADD command: ADD doc_id [vector]"""
        try:
            if len(args) < 2:
                return "❌ Usage: ADD doc_id [vector_as_json_array]"
            
            doc_id = args[0]
            
            # Join the remaining parts and parse as JSON
            vector_str = ' '.join(args[1:])
            
            # Try to parse as JSON array
            try:
                vector = json.loads(vector_str)
                if not isinstance(vector, list):
                    return "❌ Vector must be a JSON array of numbers"
            except json.JSONDecodeError:
                return "❌ Invalid JSON format for vector. Use: [1.0, 2.0, 3.0]"
            
            # Validate vector contains only numbers
            try:
                vector = [float(x) for x in vector]
            except (ValueError, TypeError):
                return "❌ Vector must contain only numeric values"
            
            # Add to database
            success = self.vectorcore.add_vector(doc_id, vector)
            
            if success:
                return f"✅ Added vector for doc_id '{doc_id}' (dimension: {len(vector)})"
            else:
                return f"❌ Failed to add vector for doc_id '{doc_id}'"
                
        except Exception as e:
            return f"❌ Error in ADD command: {str(e)}"
    
    def handle_query_command(self, args: List[str]) -> str:
        """Handle QUERY command: QUERY k [vector]"""
        try:
            if len(args) < 2:
                return "❌ Usage: QUERY k [vector_as_json_array]"
            
            try:
                k = int(args[0])
                if k <= 0:
                    return "❌ k must be a positive integer"
            except ValueError:
                return "❌ k must be an integer"
            
            # Join the remaining parts and parse as JSON
            vector_str = ' '.join(args[1:])
            
            try:
                query_vector = json.loads(vector_str)
                if not isinstance(query_vector, list):
                    return "❌ Query vector must be a JSON array of numbers"
            except json.JSONDecodeError:
                return "❌ Invalid JSON format for query vector. Use: [1.0, 2.0, 3.0]"
            
            # Validate vector contains only numbers
            try:
                query_vector = [float(x) for x in query_vector]
            except (ValueError, TypeError):
                return "❌ Query vector must contain only numeric values"
            
            # Query the database
            results = self.vectorcore.query_similar_with_distances(query_vector, k)
            
            if not results:
                return "📭 No similar vectors found (database might be empty)"
            
            # Format results
            response = f"🔍 Found {len(results)} similar vectors:\n"
            for i, (doc_id, distance) in enumerate(results, 1):
                response += f"  {i}. {doc_id} (distance: {distance:.6f})\n"
            
            # Also return just the doc_ids as requested in the spec
            doc_ids = [doc_id for doc_id, _ in results]
            response += f"\n📋 Doc IDs: {doc_ids}"
            
            return response
            
        except Exception as e:
            return f"❌ Error in QUERY command: {str(e)}"
    
    def handle_stats_command(self) -> str:
        """Handle STATS command."""
        try:
            stats = self.vectorcore.get_stats()
            
            response = "📊 VectorCore Database Statistics:\n"
            response += f"  📄 Vector count: {stats['vector_count']}\n"
            response += f"  📐 Dimension: {stats['dimension']}\n"
            response += f"  ⏰ Uptime: {stats['uptime_seconds']}s\n"
            response += f"  🌳 K-D Tree size: {stats['kdtree_size']}\n"
            response += f"  👥 Active clients: {self.client_count}"
            
            return response
            
        except Exception as e:
            return f"❌ Error getting stats: {str(e)}"
    
    def handle_save_command(self, filename: str) -> str:
        """Handle SAVE command."""
        try:
            success = self.vectorcore.save_to_disk(filename)
            if success:
                return f"💾 Database saved to '{filename}'"
            else:
                return f"❌ Failed to save database to '{filename}'"
        except Exception as e:
            return f"❌ Error saving database: {str(e)}"
    
    def handle_load_command(self, filename: str) -> str:
        """Handle LOAD command."""
        try:
            success = self.vectorcore.load_from_disk(filename)
            if success:
                stats = self.vectorcore.get_stats()
                return f"📂 Database loaded from '{filename}' ({stats['vector_count']} vectors)"
            else:
                return f"❌ Failed to load database from '{filename}'"
        except Exception as e:
            return f"❌ Error loading database: {str(e)}"
    
    def handle_clear_command(self) -> str:
        """Handle CLEAR command."""
        try:
            success = self.vectorcore.clear_all()
            if success:
                return "🗑️  Database cleared"
            else:
                return "❌ Failed to clear database"
        except Exception as e:
            return f"❌ Error clearing database: {str(e)}"
    
    def handle_get_command(self, doc_id: str) -> str:
        """Handle GET command."""
        try:
            vector = self.vectorcore.get_vector(doc_id)
            if vector is not None:
                return f"📄 Vector for '{doc_id}': {vector.tolist()}"
            else:
                return f"❌ No vector found for doc_id '{doc_id}'"
        except Exception as e:
            return f"❌ Error getting vector: {str(e)}"
    
    def handle_remove_command(self, doc_id: str) -> str:
        """Handle REMOVE command."""
        try:
            success = self.vectorcore.remove_vector(doc_id)
            if success:
                return f"🗑️  Removed vector for doc_id '{doc_id}'"
            else:
                return f"❌ No vector found for doc_id '{doc_id}'"
        except Exception as e:
            return f"❌ Error removing vector: {str(e)}"
    
    def get_help_text(self) -> str:
        """Return help text with available commands."""
        return """
🎯 VectorCore Database Commands:

📥 ADD doc_id [vector]
   Add a vector to the database
   Example: ADD doc_123 [0.1, 0.2, 0.3]

🔍 QUERY k [vector]
   Find k most similar vectors
   Example: QUERY 5 [0.1, 0.2, 0.3]

📊 STATS
   Get database statistics

💾 SAVE [filename]
   Save database to disk (default: vectorcore_data.pkl)

📂 LOAD [filename]
   Load database from disk (default: vectorcore_data.pkl)

📄 GET doc_id
   Get a specific vector by doc_id

🗑️  REMOVE doc_id
   Remove a vector from the database

🗑️  CLEAR
   Clear all vectors from database

❓ HELP
   Show this help message

📝 Note: Vectors should be JSON arrays like [1.0, 2.0, 3.0]
        """


if __name__ == "__main__":
    server = VectorCoreServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server.stop() 