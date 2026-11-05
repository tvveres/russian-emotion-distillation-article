from pathlib import Path
import urllib.request,hashlib
url='https://raw.githubusercontent.com/sismetanin/rureviews/2f6e61d30dc6c58ba47dce72e5803e1f0a3f38a7/women-clothing-accessories.3-class.balanced.csv'
p=Path('reviews.csv')
if not p.exists():urllib.request.urlretrieve(url,p)
expected='db226708d966a1506560136ef85b33ea326eaf6119465d4b1edb5bf9395330fd'
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,'Исходный файл отличается от использованного в эксперименте'
print('RuReviews: SHA-256 подтверждён')
