# coding = UTF-8
# NeruIPS 2019 论文爬取

import urllib.request
import re
import os


# open the url and read
def getHtml(url):
    page = urllib.request.urlopen(url)
    html = page.read()
    page.close()
    return html


# compile the regular expressions and find
# all stuff we need
def getUrl(html):
    reg = r'\"/paper/.*?"'
    url_re = re.compile(reg)
    url_lst = url_re.findall(html.decode('UTF-8'))
    return (url_lst)


def getFile(url):
    file_name = url.split('/')[-1]
    u = urllib.request.urlopen(url)
    f = open(file_name, 'wb')

    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        f.write(buffer)
    f.close()
    print("Sucessful to download" + " " + file_name)


root_url = 'http://papers.nips.cc'

raw_url = 'http://papers.nips.cc/book/advances-in-neural-information-processing-systems-32-2019'

html = getHtml(raw_url)
url_lst = getUrl(html)

root_dir = '/home/skylark/PycharmRemote/ldf_download'
if not os.path.exists(root_dir):
    os.mkdir('/home/skylark/PycharmRemote/ldf_download')
os.chdir('/home/skylark/PycharmRemote/ldf_download')

for url in url_lst[:]:
    url = root_url + url.split('"')[1] + '.pdf'
    getFile(url)
