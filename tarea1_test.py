from tarea1 import mapping
from tarea1.mapping import animals

if __name__ == '__main__':
    for (i, item) in enumerate(animals):
        print(f"{i}->'{item}'")
