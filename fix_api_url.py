import os
import re

directory = r"C:\Users\khush\OneDrive\Desktop\Jagrit\content-ranking-engine\web\src"

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    if 'http://localhost:5000' not in content and 'API_URL' not in content:
        return
        
    print(f"Fixing {filepath}")
    
    # 1. Replace single-quoted hardcoded urls: 'http://localhost:5000/something' -> `${API_URL}/something`
    content = re.sub(r"'http://localhost:5000(.*?)'", r"`${API_URL}\1`", content)
    
    # 2. Replace already backticked ones: `http://localhost:5000/something` -> `${API_URL}/something`
    content = content.replace("http://localhost:5000", "${API_URL}")
    
    # 3. Add the import definition if not present
    if "const API_URL" not in content:
        import_index = content.rfind("import ")
        if import_index != -1:
            next_newline = content.find("\n", import_index)
            inject_str = "\nconst API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';\n"
            content = content[:next_newline+1] + inject_str + content[next_newline+1:]
            
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith(('.js', '.jsx')):
            fix_file(os.path.join(root, file))
            
print("Fixed API_URLs safely!")
