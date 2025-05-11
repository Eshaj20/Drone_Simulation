from twilio.rest import Client

# Twilio API setup (store your credentials in a config file or environment variables)
def send_sms_alert(message):
    account_sid = 'account_sid'
    auth_token = 'auth_token'
    twilio_phone = 'twilio_phone_number'
    recipient_phone = 'recipient_phone_number'
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=message,
        from_=twilio_phone,
        to=recipient_phone
    )
    print(f"SMS Alert sent: {message.sid}")
