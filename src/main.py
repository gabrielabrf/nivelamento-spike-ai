import sys
from pathlib import Path

# Adiciona a raiz do projeto ao path para evitar erros de importação
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.redender import render

def main():
    render()

if __name__ == "__main__":
    main()