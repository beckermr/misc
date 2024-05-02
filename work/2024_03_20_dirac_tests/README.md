# instructions

## register with gridpp via ciologon

1. Go to cilogon.org and generate a cetificate and download (it needs to be silver or higher under cert info)
2. Add to keychain on osx by opening the app and dropping the downloaded certificate into the keychain app
4. Go to this page in firefox: https://voms.gridpp.ac.uk:8443/voms/gridpp/register/start.action
   Firefox will complain about things not being secure but click through to ignore that

## make .pem from .p12

```bash
openssl pkcs12 -in usercred.p12 -out usercert.pem -nodes -legacy
```
