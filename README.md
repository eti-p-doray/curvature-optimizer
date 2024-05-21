
Before setting up, this needs to run for each new shell session:

`module restore jax_modules`
`./init.sh`

To create and setup a new venv:

`virtualenv --no-download .env`
`source .env/bin/activate`
`pip install --no-index --upgrade pip`
`pip install --no-index -r requirements.txt`

See https://docs.alliancecan.ca/wiki/Python

Subsequent shell sessions need to activate the venv:
`source .env/bin/activate`