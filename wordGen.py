import requests
import time
import random

word_site = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"

response = requests.get(word_site)
WORDS = response.content.splitlines()

random.shuffle(WORDS)

for i in WORDS:
    print(str(i))
    time.sleep(1.5)
