### Flight Fare Prediction

#

### Software and Tools Requirement

1. [Github Account](https://github.com)
2. [VS Code IDE](https://code.visualstudio.com/download)
3. [Git CLI](https://git-scm.com/downloads)
4. [AWS](https://signin.aws.amazon.com/signin?redirect_uri=https%3A%2F%2Fconsole.aws.amazon.com%2Fconsole%2Fhome%3FhashArgs%3D%2523%26isauthcode%3Dtrue%26nc2%3Dh_ct%26src%3Dheader-signin%26state%3DhashArgsFromTB_ap-northeast-1_ef2524f0af9cef74&client_id=arn%3Aaws%3Asignin%3A%3A%3Aconsole%2Fcanvas&forceMobileApp=0&code_challenge=Sfrl3h4obawgO2thMln3jNbGpGNgNLfI42oub-ve2SY&code_challenge_method=SHA-256)

#

### Website Link and Screenshots
[Click here](http://predictflightfare-env.eba-438dqfva.ap-south-1.elasticbeanstalk.com/) to visit the Deployed prediction web page

Home Page
https://github.com/tinkuhore/flight_fare_prediction/issues/1#issue-1551210745

Prediction result for all available Airlines
https://github.com/tinkuhore/flight_fare_prediction/issues/1#issuecomment-1398718154

Prediction result for any particular Airline
https://github.com/tinkuhore/flight_fare_prediction/issues/1#issuecomment-1398718535

#

### Create a new environment for the project

Using anaconda
```
conda create -p venv python==3.7 -y
```

If conda not installed
```
python3 -m venv .venv
```
To acivate the environment
```
source .venv/bin/activate
```
#

### Git commands

Configuration
```
git config --global user.name "<your name>"
git config --global user.email "<mail id registered with github>"
```
Check status
```
git status
```
Check logs
```
git log
```
Add all new or modified files
```
git add .
```
Add or update specific file
```
git add <file name>
```
Commit changes
```
git commit -m "<commit message>"
```
Push the commited changes
```
git push <remote> <branch>
```
#

### Deployment Steps

##### The WebApp was deployed to AWS Elastic Beanstalk


- Create a new folder named ".ebextensions"

- In that folder create a file <any_name>.config (* Make sure the extension is .config)

- Above file should contain the following
  ```
  option_settings:
    "aws:elasticbeanstalk:container:python":
      WSGIPath: <main python file name>:<flask app name>
  ```

- Make a zip file with all necessary files and folder

- Login to AWS

- Search for *Elastic Beanstalk* and click *Create Application*

- Give any valid name, select platform and upload the zip file. Finally click on **Create Application**.

If you are unable to follow the above steps properly watch this [VIDEO](https://www.youtube.com/watch?v=zn23teIOcVw)

#


### **CI/CD Pipeline** using **GitHub actions** for the application hosted via AWS Elastic BeanStalk


We have already deployed our Flask App and got the [url](http://predictflightfare-env.eba-438dqfva.ap-south-1.elasticbeanstalk.com/)

Now, we need to develop a CI/CD Pipeline so that when ever we push a new update for our App, it will automatically get deployed and reflect those changes on the same url.

Steps are : 

- Create AWS IAM User with any name (say git actions) with AWSS3 and AWSEB full permissions.

- Download new_user_credentials.csv to get the required secrets.

- In git repo go to settings -> secrets -> actions -> New Repo Secrets

- Save secrets named AWS_CCESS_KEY_ID and AWS_SECRET_ACCESS_KEY with respective values from the downloaded csv file.

- Create a S3 Bucket with same AWS Region.

- Create <any_name>.yml file in the folder .github/workflows/.

- Mention all the actions to be executed with a particular trigger (say git push)
