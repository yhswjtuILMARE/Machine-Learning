'''
Created By ILMARE
@Date 2019-3-1
'''

from bs4 import BeautifulSoup
from urllib.request import urlretrieve
import requests
import os

def get_title_list(pageNum=0):
    Url = "http://tieba.baidu.com/f?kw=%E6%9D%A8%E5%B9%82&ie=utf-8&pn={0}".format(pageNum * 50)
    try:
        resp = requests.get(Url)
        bsObj = BeautifulSoup(resp.text, "html.parser")
        elts = bsObj.find_all("li", {"class": "j_thread_list clearfix"})
        return_mat = []
        for elt in elts:
            repNum = int(elt.find("span", {"class": "threadlist_rep_num center_text"}).text)
            a = elt.find("a", {"class": "j_th_tit"})
            link = a.attrs.get("href")
            title = a.attrs.get("title")
            return_mat.append((title, "{0}{1}".format("http://tieba.baidu.com", link), repNum))
        return return_mat
    except Exception as e:
        print(e)
        return None

if __name__ == "__main__":
    Url = "http://tieba.baidu.com/p/6051406866"
    try:
        resp = requests.get(Url)
        bsObj = BeautifulSoup(resp.text, "html.parser")
        ul = bsObj.find("ul", {"class": "l_posts_num"})
        totalPage = int(ul.find("li", {"class": "l_reply_num"}).find_all("span", {"class": "red"})[1].text)
        elts = bsObj.find_all("div", {"class": ["l_post", "j_l_post", "l_post_bright", "noborder"]})
        for elt in elts:
            div = elt.find("div", {"class": "d_post_content j_d_post_content clearfix"})
            print(div.find_all("img"))
        print(len(elts))
    except Exception as e:
        print(e)

