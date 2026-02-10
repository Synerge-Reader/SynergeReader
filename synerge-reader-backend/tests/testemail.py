import resend
from dotenv import load_dotenv
import os
load_dotenv()
resend.api_key = os.getenv("EMAIL_KEY")


params = {
    "from": "Synerge <no-reply@synergereader.ai>",
    "to": ["_@gmail.com"],
    "subject": "Blah",
    "html": "<p>yh</p>",
}

email = resend.Emails.send(params)
print(email)