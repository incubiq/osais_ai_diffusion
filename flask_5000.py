##
## when in docker:
##
##   Put yourself in the JUST-PING directory
##   Go in Linux : wsl
##   Build the docker image: docker build -t yeepeekoo/my_images:ai_ping .
##   Run doker instance: docker run -it -v "$(pwd)/_input:/src/app/_input" -v "$(pwd)/_output:/src/app/_output" --name ai_ping --rm --gpus all --publish 5001:5000 yeepeekoo/my_images:ai_ping
##   Test it: http://localhost:5001/run?orig=TODO&username=123&uid=4343&token=123098&size=800&output=test.jpg&input=622969762353.jpg
##
##
## when debug locally:
##
##   Put yourself in the JUST-PING directory
##   Set env var once:  $env:FLASK_APP="flask_5000"
##   Run with: python -m flask run --host=0.0.0.0 --port=5000
##   Test it : http://localhost:5000/run?username=123&uid=4343&token=123098&size=800&output=test.jpg&input=..%2F_uploads%2F551918561430.jpg&odir=.%2F_output%2F
##

## ------------------------------------------------------------------------
#       connect the AI with OSAIS
## ------------------------------------------------------------------------

import sys
#from osais_debug import osais_initializeAI, osais_getInfo, osais_getHarwareInfo, osais_getDirectoryListing, osais_runAI, osais_authenticateAI, osais_isLocal
from osais import osais_initializeAI, osais_getInfo, osais_getHarwareInfo, osais_getDirectoryListing, osais_runAI, osais_authenticateAI, osais_isLocal

## register and login this AI
APP_ENGINE=osais_initializeAI()
if APP_ENGINE==None:
    sys.exit(0)

## ------------------------------------------------------------------------
#       AI endpoint and warmup
## ------------------------------------------------------------------------

import time
from flask import Flask, request, jsonify, render_template
app = Flask(
    APP_ENGINE,  
    template_folder='_template',
    static_folder='_output')

import json
import sys
sys.path.insert(0, './scripts')

# globals (to avoid cost of re-init each time)
gModel=0
gDevice=0

def getArgs(_args):
    result = []
    for key, value in _args.items():
        if key.startswith("-"):
            result.append(key)
            result.append(value)
    return result

## ------------------------------------------------------------------------
#       routes for this AI (important ones)
## ------------------------------------------------------------------------

@app.route('/')
def home():
    return jsonify({"data":osais_getInfo()})

@app.route('/auth')
def auth():
    return jsonify({"data": osais_authenticateAI()})

@app.route('/status')
def status():
    return jsonify({"data": osais_getInfo()})

@app.route('/run')
def run():
    from img2img import fn_runImg
    from txt2img import fn_runTxt

    aFinalArg=getArgs(request.args)        
    print("processed args from url: "+str(aFinalArg))
    
    global gModel
    global gDevice

    if request.args.get('-filename')==None:
        print("Running TXT2IMG")
        aOutput, model, device= osais_runAI(fn_runTxt, request.args, gModel, gDevice)
        # aOutput, model, device=fn_runTxt(aFinalArg, gModel, gDevice)
    else:
        print("Running IMG2IMG")
        aOutput, model, device= osais_runAI(fn_runImg, request.args, gModel, gDevice)
        # aOutput, model, device=fn_runImg(aFinalArg, gModel, gDevice)

    gModel=model
    gDevice=device
    return showManualPage(request.args, aOutput)

def showManualPage(args, aOutput):
    return render_template('./image_result.html', 
        files=aOutput, 
        prompt=args.get('--prompt'), 
        seed=args.get('--seed'), 
        strength=args.get('--strength'),
        steps=args.get('--steps'),
        W=args.get('--W'),
        H=args.get('--H'),
        cimg=args.get('--n_samples'),

        username=args.get('-username'),
        token=args.get('-t'),
        uid=args.get('-uid'),

        orig=args.get('-orig'),
        odir=args.get('-odir'),
        idir=args.get('-idir'),
        output=args.get('--output'),
        input=args.get('-filename'))

## ------------------------------------------------------------------------
#       routes for this AI (optional)
## ------------------------------------------------------------------------

@app.route('/gpu')
def gpu():
    return jsonify(osais_getHarwareInfo())

@app.route('/manual')
def manual():
    ts=int(time.time())
    return render_template('./image_result.html', 
        prompt="zombie", 
        seed=100, 
        strength=0.5,
        steps=20,
        cimg=2,
        files=[],
        W=512,
        H=512,

        username="http://192.168.1.83:3022",
        token="a641d6413a99f8fe50a28f31b456af7ccc38cd34baac87e5f978d140bb0e1fc2",
        uid=str(ts),
        
        odir="D:\\Websites\\opensourceais\\backend_public\\_temp\\output",
        idir="D:\\Websites\\opensourceais\\backend_public\\_temp\\input",
        
        orig =  'http://192.168.1.83:3022',

        output="result",
        input="clown.jpg")

@app.route('/test')
def test():
    from werkzeug.datastructures import MultiDict
    ts=int(time.time())
    seed=int(time.time())

    sample_args = MultiDict([
        ('-u', 'http://192.168.1.83:3022'),
        ('-uid', str(ts)),
        ('-t', 'a641d6413a99f8fe50a28f31b456af7ccc38cd34baac87e5f978d140bb0e1fc2'),
        ('-local', 'True'),
        ('--W', '512'),
        ('--H', '512'),
        ('--prompt', 'zombie'),
        ('--seed', '100'),
        ('--strength', '0.5'),
        ('--seed', str(seed)),
        ('--ddim_steps', '20'),
        ('--n_samples', '2'),
        ('--output', str(ts)+'.jpg'),
        ('-idir', 'D:\\Websites\\opensourceais\\backend_public\\_temp\\input'),
        ('-odir', 'D:\\Websites\\opensourceais\\backend_public\\_temp\\output'),
        ('-orig', 'http://192.168.1.83:3022/'),
    ])

    isImg2Img=True
    if isImg2Img:
        sample_args.add('-filename', 'clown.jpg')

    global gModel
    global gDevice

    if isImg2Img==False:
        from txt2img import fn_runTxt
        aOutput, model, device= osais_runAI(fn_runTxt, sample_args, gModel, gDevice)
    else:
        from img2img import fn_runImg
        aOutput, model, device= osais_runAI(fn_runImg, sample_args, gModel, gDevice)

    gModel=model
    gDevice=device

    return showManualPage(sample_args, aOutput)


## ------------------------------------------------------------------------
#       test routes when in local mode
## ------------------------------------------------------------------------

if osais_isLocal():
    @app.route('/root')
    def root():
        return osais_getDirectoryListing("./")

    @app.route('/input')
    def input():
        return osais_getDirectoryListing("./_input")

    @app.route('/output')
    def output():
        return osais_getDirectoryListing("./_output")

