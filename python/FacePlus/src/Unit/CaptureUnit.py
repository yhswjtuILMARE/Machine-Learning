'''
Created on 2018年7月5日

@author: IL MARE
'''

import requests

def getImage():
    try:
        '''
        2465
        3270
        '''
        for str in "06":
            if str == "0":
                index = 2465
            else:
                index = 3270
            for i in range(1, index + 1):
                hostName = "http://yjsjy.nwpu.edu.cn/pyxx/grxx/xszphd/zp/ykt/20172{0}{1:04d}".format(str, i)
                cookie = {
                    "JSESSIONID": "jZkQb2ZBpn1KXVw4vlsyqBYgHLF8LghnSVdV1FMBsC1PhH8YVqXv!1232870148",
                    }
                resp = requests.get(hostName, timeout=15, cookies=cookie)
                try:
                    fp = open(r"G:/python/sources/nwpu/image-2{0}/{1}.jpg".format(str, "20172{0}{1:04d}".format(str, i)), "wb")
                    print(r"G:/python/sources/nwpu/image-2{0}/{1}.jpg".format(str, "20172{0}{1:04d}".format(str, i)))
                    fp.write(resp.content)
                except Exception as e:
                    print(e)
                finally:
                    fp.close()
    except Exception as e:
        print(e)    

if __name__ == "__main__":
    pass