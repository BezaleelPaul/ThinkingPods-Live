with open('measure_latency.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace common Unicode chars with ASCII
content = content.replace('\u2713', '[OK]')
content = content.replace('\u2717', '[FAIL]')
content = content.replace('\u2014', '--')
content = content.replace('\u201c', '"')
content = content.replace('\u201d', '"')
content = content.replace('\u2018', "'")
content = content.replace('\u2019', "'")
content = content.replace('\u2026', '...')
content = content.replace('\u2022', '-')
content = content.replace('\u2013', '-')
content = content.replace('\u00d7', 'x')

with open('measure_latency.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed Unicode characters')