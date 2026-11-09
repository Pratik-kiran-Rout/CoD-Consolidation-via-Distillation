import json

lines = open('C:/Users/ASUS/.gemini/antigravity/brain/991582e7-f38b-41a5-8f1a-18c51379bea9/.system_generated/logs/transcript_full.jsonl', 'r', encoding='utf-8').readlines()
for i in range(len(lines)-1, -1, -1):
    obj = json.loads(lines[i])
    if obj.get('source') == 'USER_EXPLICIT':
        print(f"Last user prompt is at line {i}, length={len(obj.get('content', ''))}")
        print(obj.get('content', '')[:200])
        # write it out to scratch/user_input_phase2.txt
        open('scratch/user_input_phase2.txt', 'w', encoding='utf-8').write(obj.get('content', ''))
        break

