from bs4 import BeautifulSoup
import requests

website = 'https://www.imot.bg/pcgi/imot.cgi?act=3&slink=86irbr&f1=1'
result = requests.get(website)
content = result.text

soup = BeautifulSoup(content, 'lxml')
print(soup.encode('utf-8'))

submenu = soup.find('div', class_='submenu')
# print(submenu)

link = submenu.find('a', class_='f_right').get_text()
# print(link)
