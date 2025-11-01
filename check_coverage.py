#!/usr/bin/env python3
import subprocess
import sys

def check_coverage():
    """Проверить покрытие кода тестами."""
    print("Проверка покрытия кода...")
    print("=" * 50)
    
    # Запускаем тесты с покрытием
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/", 
        "--cov=src", 
        "--cov-report=term"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    
    if result.stderr:
        print("Ошибки:")
        print(result.stderr)

if __name__ == "__main__":
    check_coverage()