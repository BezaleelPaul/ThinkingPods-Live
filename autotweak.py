import subprocess
import os
import sys

def run_command(cmd):
    try:
        # Run with check=True to raise exception on error, and capture outputs to keep terminal clean
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}\nError: {e.stderr.decode().strip()}")
        return False

def get_physical_cores():
    """ Determines physical core count on macOS, Windows, or Linux. """
    # macOS check
    if sys.platform == "darwin":
        try:
            # Query Performance Cores first (preferred for Apple Silicon matrix ops)
            output = subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).decode().strip()
            if output:
                return int(output)
        except Exception:
            pass
        try:
            output = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip()
            if output:
                return int(output)
        except Exception:
            pass
            
    # Windows check
    elif sys.platform == "win32":
        try:
            output = subprocess.check_output('powershell -Command "(Get-CimInstance Win32_Processor).NumberOfCores"', shell=True).decode().strip()
            if output:
                return int(output)
        except Exception:
            pass
        try:
            output = subprocess.check_output("wmic cpu get NumberOfCores", shell=True).decode().strip()
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            if len(lines) > 1:
                return int(lines[1])
        except Exception:
            pass
            
    # Linux check
    else:
        try:
            output = subprocess.check_output("lscpu -p=Core | grep -v '^#' | sort -u | wc -l", shell=True).decode().strip()
            if output:
                return int(output)
        except Exception:
            pass
    
    # Generic Fallback: using multiprocessing to get logical core count
    import multiprocessing
    logical = multiprocessing.cpu_count()
    # Consumer CPUs typically have 2 threads per core
    return max(1, logical // 2)

def get_total_ram_gb():
    """ Determines total physical memory size in GB on macOS, Windows, or Linux. """
    # macOS check
    if sys.platform == "darwin":
        try:
            output = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            if output:
                return int(output) / (1024**3)
        except Exception:
            pass
            
    # Windows check
    elif sys.platform == "win32":
        try:
            output = subprocess.check_output('powershell -Command "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"', shell=True).decode().strip()
            if output:
                return int(output) / (1024**3)
        except Exception:
            pass
        try:
            output = subprocess.check_output("wmic ComputerSystem get TotalPhysicalMemory", shell=True).decode().strip()
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            if len(lines) > 1:
                bytes_val = int(lines[1])
                return bytes_val / (1024**3)
        except Exception:
            pass
            
    # Linux check
    else:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        parts = line.split()
                        return int(parts[1]) / (1024**2) # /proc/meminfo is in KB, convert to GB
        except Exception:
            pass
            
    return 8.0  # Safe default fallback

def autotweak():
    print("\n==============================================")
    print("      DYNAMIC HARDWARE PROFILER & AUTO-TWEAKER")
    print("==============================================")
    print("Profiling system hardware specifications...")
    
    physical_cores = get_physical_cores()
    total_ram = get_total_ram_gb()
    
    print(f" -> Detected Physical Cores: {physical_cores}")
    print(f" -> Detected System RAM: {total_ram:.2f} GB")
    
    # 1. The "Physical Core" Rule & 2. The 8-Thread Bandwidth Cap
    # For CPU inference, threading beyond physical cores or past memory bus limits (max 8) slows down execution.
    optimal_threads = min(physical_cores, 8)
    
    # 4. Aggressive Context Truncation
    # If the system has < 8GB RAM, cap context aggressively to 1024. Otherwise, use 1536 (or 2048 max).
    optimal_ctx = 1024 if total_ram < 8.0 else 1536
    
    print(f" -> Calibrated Thread Allocation: {optimal_threads} threads (Physical Core & Memory Bandwidth Cap)")
    print(f" -> Calibrated Context Size: {optimal_ctx} tokens (Aggressive Context Truncation)")
    
    base_model = os.getenv("MENTOR_MODEL", "qwen2.5:3b")
    
    # Check if Ollama service is reachable and if base model is available
    print("\nVerifying local Ollama service connection...")
    try:
        output = subprocess.check_output("ollama list", shell=True).decode()
        if base_model not in output:
            print(f" -> Base model '{base_model}' is not currently pulled.")
            print(f" -> Attempting to pull '{base_model}' now (requires internet connection)...")
            # Try to pull, but catch failure if offline
            pulled = run_command(f"ollama pull {base_model}")
            if not pulled:
                print(f" -> [WARNING] Could not pull model automatically (likely offline).")
                print(f"    Please ensure you have pulled it: 'ollama pull {base_model}'")
        else:
            print(f" -> Verified: Base model '{base_model}' is ready.")
    except Exception as e:
        print(" -> [ERROR] Could not connect to Ollama service. Please make sure Ollama is running ('ollama serve').")
        return False
        
    # Write optimal parameters into a temporary Modelfile
    modelfile_content = f"""FROM {base_model}
PARAMETER num_ctx {optimal_ctx}
PARAMETER num_thread {optimal_threads}
PARAMETER f16_kv false
PARAMETER temperature 0.7
"""
    
    modelfile_path = "Modelfile.dynamic"
    try:
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        print(f" -> Compiled optimal container profile into '{modelfile_path}'")
        
        print(" -> Compiling optimized-pods model container in Ollama on the fly...")
        success = run_command(f"ollama create optimized-pods -f {modelfile_path}")
        
        if success:
            print("[SUCCESS] Successfully built and compiled 'optimized-pods' model!")
            print("==============================================\n")
            return True
        else:
            print("[ERROR] Failed to compile optimized-pods container.")
            return False
    except Exception as e:
        print(f" -> [ERROR] Failed during Modelfile creation or compilation: {e}")
        return False
    finally:
        # Clean up temporary Modelfile
        if os.path.exists(modelfile_path):
            try:
                os.remove(modelfile_path)
            except OSError:
                pass

if __name__ == "__main__":
    autotweak()
