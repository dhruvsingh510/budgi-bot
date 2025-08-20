import os
from pathlib import Path
from datetime import datetime, timedelta
import pytz

class LogViewer:
    """Utility to view and filter service logs."""
    
    def __init__(self):
        self.logs_dir = Path(__file__).parent.parent / "data" / "logs"
        self.ist_tz = pytz.timezone('Asia/Kolkata')
    
    def view_service_logs(self, service_name: str, lines: int = 50):
        """View recent logs for a specific service."""
        log_file = self.logs_dir / f"{service_name}.log"
        
        if not log_file.exists():
            print(f"❌ Log file for {service_name} not found: {log_file}")
            return
        
        print(f"📋 Last {lines} lines from {service_name} service:")
        print("=" * 80)
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
                for line in recent_lines:
                    print(line.rstrip())
                    
        except Exception as e:
            print(f"❌ Error reading log file: {e}")
    
    def view_combined_logs(self, lines: int = 100):
        """View recent logs from all services combined."""
        log_file = self.logs_dir / "combined.log"
        
        if not log_file.exists():
            print(f"❌ Combined log file not found: {log_file}")
            return
        
        print(f"📋 Last {lines} lines from all services:")
        print("=" * 80)
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
                for line in recent_lines:
                    print(line.rstrip())
                    
        except Exception as e:
            print(f"❌ Error reading log file: {e}")
    
    def filter_logs_by_time(self, service_name: str, hours_back: int = 1):
        """Filter logs by time (last N hours)."""
        log_file = self.logs_dir / f"{service_name}.log"
        
        if not log_file.exists():
            print(f"❌ Log file for {service_name} not found")
            return
        
        cutoff_time = datetime.now(self.ist_tz) - timedelta(hours=hours_back)
        cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"📋 {service_name} logs from last {hours_back} hour(s) (since {cutoff_str} IST):")
        print("=" * 80)
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if cutoff_str <= line[:19]:  # Compare timestamp part
                        print(line.rstrip())
                        
        except Exception as e:
            print(f"❌ Error filtering logs: {e}")
    
    def search_logs(self, service_name: str, search_term: str, lines: int = 20):
        """Search for specific terms in logs."""
        log_file = self.logs_dir / f"{service_name}.log"
        
        if not log_file.exists():
            print(f"❌ Log file for {service_name} not found")
            return
        
        print(f"🔍 Searching for '{search_term}' in {service_name} logs:")
        print("=" * 80)
        
        try:
            matches = []
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if search_term.lower() in line.lower():
                        matches.append((line_num, line.rstrip()))
            
            if not matches:
                print(f"No matches found for '{search_term}'")
                return
            
            # Show recent matches
            recent_matches = matches[-lines:] if len(matches) > lines else matches
            
            for line_num, line in recent_matches:
                print(f"Line {line_num}: {line}")
                
            print(f"\nFound {len(matches)} total matches")
                        
        except Exception as e:
            print(f"❌ Error searching logs: {e}")

def main():
    """CLI interface for log viewer."""
    import sys
    
    viewer = LogViewer()
    
    if len(sys.argv) < 2:
        print("📋 Log Viewer Usage:")
        print("  python log_viewer.py view <service> [lines]")
        print("  python log_viewer.py combined [lines]")
        print("  python log_viewer.py filter <service> <hours>")
        print("  python log_viewer.py search <service> <term>")
        print()
        print("Available services: orchestrator, budget, transaction")
        return
    
    command = sys.argv[1]
    
    if command == "view" and len(sys.argv) >= 3:
        service = sys.argv[2]
        lines = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        viewer.view_service_logs(service, lines)
        
    elif command == "combined":
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        viewer.view_combined_logs(lines)
        
    elif command == "filter" and len(sys.argv) >= 4:
        service = sys.argv[2]
        hours = int(sys.argv[3])
        viewer.filter_logs_by_time(service, hours)
        
    elif command == "search" and len(sys.argv) >= 4:
        service = sys.argv[2]
        term = sys.argv[3]
        viewer.search_logs(service, term)
        
    else:
        print("❌ Invalid command or missing arguments")

if __name__ == "__main__":
    main()