import requests
import time
import random

# Website to generate dictionary of words
word_site = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
response = requests.get(word_site)
WORDS = response.content.splitlines()

# Shuffle words
random.shuffle(WORDS)

# Display new word every 1.5 seconds
for i in WORDS:
    print(str(i))
    time.sleep(1.5)
