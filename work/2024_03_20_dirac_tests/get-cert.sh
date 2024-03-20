echo -n \
    | openssl s_client -connect dirac.gridpp.ac.uk:8443 -servername dirac.gridpp.ac.uk \
    | openssl x509 > certificate.cert
