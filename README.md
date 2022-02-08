# ri_digitaltwin
Thesis research project, that involves a microsimulation in Roosevelt Island New York

#installation 
(based on this:
https://github.com/techwithtim/Flask-App-Hosted-On-VPS
)
1. Install a new Ubuntu 20.4 (LTR) in Linode
2. install the following
```
sudo apt-get update
sudo apt-get install python3
sudo apt-get install python3-pip
sudo apt-get git
pip3 install -r requirements.txt
```

3. Install nginx and create a new configuration file.
```
sudo apt install nginx 
sudo nano /etc/nginx/sites-enabled/<directory-name-of-flask-app>
```

4. The contents of the confiugration file should be as follows:

```
server {
    listen 80;
    server_name <public-server-ip>;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```
5. Unlink the default config file and reload nginx to use the newly created config file.
```
sudo unlink /etc/nginx/sites-enabled/default
sudo nginx -s reload
```

6. install unicorn
sudo apt-get install gunicorn

7. Run the flask web app with gunicorn. The name is the filename and the app is the object -t is the timeout

```
gunicorn -w 3 flask_app:app -t 180
```