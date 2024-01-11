import urllib.request
from urllib.parse import quote
import httplib2
import json
from tqdm import tqdm
from pathlib import Path
import hashlib

API_KEY = ""
CUSTOM_SEARCH_ENGINE = ""
keywords = ["kiwi flower", "キウイ 花"]
img_list = []


def get_image_url(keyword, num):
    for i in tqdm(range(0, num // 10 + 1)):
        query_img = "https://www.googleapis.com/customsearch/v1?key=" + API_KEY + "&cx=" + CUSTOM_SEARCH_ENGINE + "&num=" + \
            str(10 if (num - i) > 10 else (num - i)) + "&start=" + str(i + 1) + "&q=" + quote(keyword) + "&searchType=image"
        res = urllib.request.urlopen(query_img)
        data = json.loads(res.read().decode('utf-8'))
        for i in range(len(data["items"])):
            img_list.append(data['items'][i]['link'])
    return img_list


def get_image(img_list):
    root_dir = Path("images")
    root_dir.mkdir(exist_ok=True)
    http = httplib2.Http(".cache")
    for i, url in enumerate(tqdm(img_list)):
        try:
            md5 = hashlib.md5(url.encode()).hexdigest()
            filename = root_dir / Path(md5 + ".jpg")
            if not filename.exists():
                _, content = http.request(url)
                with open(filename, 'wb') as f:
                    f.write(content)
        except Exception as e:
            print("failed to download the image.", e)
            continue


def main():
    img_list = []
    for i in range(len(keywords)):
        img_list.extend(get_image_url(keywords[i], 300))

    img_list = list(set(img_list))
    get_image(img_list)


if __name__ == '__main__':
    main()
