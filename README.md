# novelgan-web

## Setup
```
$ git clone git@github.com:tayutaedomo/novelgan-web.git
$ cd novelgan-web
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```


## git lfs
You have to execute the following commands as use git lfs.
```
$ git lfs install
# git lfs track "*.h5"
```


## Local Server
```
$ cd novelgan-web
$ source venv/bin/activate
$ python app.py
$ open 'http://127.0.0.1:5000/'
```
Basic Auth: novelgan/novels


## Docker
```
$ cd novelgan-web
$ docker build -t novelgan-web .
$ docker run --rm -it -e PORT=8080 -p 8080:8080 novelgan-web
$ open 'http://0.0.0.0:8080'
```


## Config ENV
You should set the appropriate ENV.
```
$ export APP_SETTINGS="config.DevelopmentConfig"
# or
$ export APP_SETTINGS=config.StagingConfig
# or
$ export APP_SETTINGS=config.ProductionConfig
```


## Cloud Run
```
$ cd novelgan-web
$ gcloud builds submit --tag gcr.io/[PROJECT-ID]/novelgan-web
$ gcloud run deploy --image gcr.io/[PROJECT-ID]/novelgan-web --platform managed
```

