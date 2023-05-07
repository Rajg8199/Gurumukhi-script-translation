with open('3.txt', encoding='utf8') as f:
    for line in f:
        print(line.strip())
        
A = line.strip()

temp = []
for ii in range(0,len(A)):
    
    temp.append(ord(A[1]))
