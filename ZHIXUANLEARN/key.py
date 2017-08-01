def md5(s):
    import hashlib
    m = hashlib.md5()
    m.update(s.encode('utf-8'))
    return  m.hexdigest()
s1 = '613BFEFA70CFB437190299B6'  #机器码


def key(s1):
    for i in range(0,30):
        s2=3*i
        s1=s1+'PDF'+str(s2)
        s1=md5(s1)
        s1=s1.upper()
    s3=s1[0:24]
    return s3
print(key(s1))