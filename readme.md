activate virtual env
```
py -m venv env
.\env\Scripts\activate
```





## generate requirements.txt
```
pip freeze > requirements.txt
```

## install package
```
pip install -r requirements.txt
```

## (optional) manual install lib
```
pip install flask
pip install -U sentence-transformers
```


# run
```
python src\app.py
```

# test
```
curl -s -H "Content-Type:application/json" -XPOST http://localhost:7700/sent_sim -d '{"prompt": "how is utilization"}'
```
