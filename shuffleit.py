# awk -F '\t' '{sum[$2]++}END{for(i in sum) print i "\t" sum[i]}' ./hello1_
# 处理数据后用shuf打乱
def test(num,line,f1,f2,f3):
    if num % 5 == 4 :
        f2.write(line)
    elif num % 5 == 0:
        f3.write(line)
    else:
        f1.write(line)
        
for i in range(5,8):
    with open('./newdata/hello'+str(i)+'_','r+') as f0:
        with open('./newdata/train'+str(i)+'.txt','a') as f1:
            with open('./newdata/valid'+str(i)+'.txt','a') as f2:
                with open('./newdata/test'+str(i)+'.txt','a') as f3:
                    a=b=c=d=1
                    e=0
                    for num,line in enumerate(f0,1):
                        if line[-2] == '1':
                            test(a,line[:-2]+str(0)+'\n',f1,f2,f3)
                            a += 1
                        elif line[-2] == '2':
                            test(b,line[:-2]+str(1)+'\n',f1,f2,f3)
                            b += 1
                        elif line[-2] == '3':
                            e += 1
                            if e % 5 ==2:
                                test(c,line[:-2]+str(2)+'\n',f1,f2,f3)
                                c += 1
                        elif line[-2] == '4':
                            test(d,line[:-2]+str(3)+'\n',f1,f2,f3)
                            d += 1
                        else:
                            pass
                            



    