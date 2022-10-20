
f = open("wifirxRes.txt", 'r')
res = f.readlines()
f.close()


f = open("sisopdr.txt", 'w')

resStr = ""
for each in res:
    if(each[0] == 'd'):
        if(resStr == ""):
            snr = each.split(':')[1]
            snr = snr[0:-1]
        else:
            resLine = ""
            resLine += snr
            resLine += " "
            resItems = resStr.split(',')
            # print(resItems)
            for i in range(0,9):
                resLine += resItems[i+3].split(':')[1]
                resLine += " "
            resLine += '\n'
            f.write(resLine)
            print(resLine)
            snr = each.split(':')[1]
            snr = snr[0:-1]
            # print(resStr)
    elif(each[0] == 'i'):
        resStr = each

f.close()

