# Welcome to streamlit

This is the app you get when you run `streamlit hello`, extracted as its own app.

Edit [Hello.py](./Hello.py) to customize this app to your heart's desire. ❤️

Check it out on [Streamlit Community Cloud](https://st-hello-app.streamlit.app/)


# Deploy to Google App Engine

`gcloud app deploy`


# Deploy to cloud run

`gcloud auth configure-docker gcr.io`

`docker build -t gcr.io/stg-transcription/video .`

`docker run -v ~/.config/gcloud/application_default_credentials.json:/root/.config/gcloud/application_default_credentials.json gcr.io/stg-transcription/video:latest`

`docker push gcr.io/stg-transcription/video`

```
gcloud run jobs create video-job \
  --image gcr.io/stg-transcription/video:latest \
  --region us-central1 \
  --task-timeout 120m \
  --max-retries 0 \
  --cpu 8 \
  --memory 32Gi
```

```
gcloud run jobs update video-job \
  --image gcr.io/stg-transcription/video:latest \
  --region us-central1 \
  --task-timeout 120m \
  --max-retries 0 \
  --cpu 8 \
  --memory 32Gi
```

```
gcloud run jobs execute video-job \
  --region=us-central1 \
  --update-env-vars ^@^LIVE_ANCHOR=false@TEST_MODE=true@REUTERS_ID=tag:reuters.com,2024:newsml_RW327824062024RP1:6 \
  --async
```

`./execute_jobs.sh false true ./reuters_ids.txt`
