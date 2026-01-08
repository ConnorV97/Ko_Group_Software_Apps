# Used for call via Labview
def read_translation_from_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read().strip()
        x_str, y_str = content.split(',')
        return float(x_str), float(y_str)


if __name__ == "__main__":
    filepath = r'C:\Users\conno\Desktop\STM Auto Files\latest_translation_flat.txt'
    x, y = read_translation_from_file(filepath)
    print(f"Read translation: x = {x} nm, y = {y} nm")