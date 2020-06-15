const tfjs = require('@tensorflow/tfjs');
const posenet = require('@tensorflow-models/posenet');
const canvas = require('canvas');

const WIDTH = 752;
const HEIGHT = 487;

let net;
const cv = canvas.createCanvas(WIDTH, HEIGHT);
const ctx = cv.getContext('2d');


async function getNet() {

  if (!net) {
    net = await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: { width: WIDTH, height: HEIGHT },
      multiplier: 0.75
    });
    
    console.log('Loaded model !');
  }

  return net;
}

async function startScript() {

  const img = await canvas.loadImage('tst.jpeg');

  console.log(img);

  ctx.drawImage(img, 0, 0, WIDTH, HEIGHT);

  net = await getNet();

  const pose = await net.estimateSinglePose(cv);

  
  //TODO: change to save to file
  console.log(pose);
}

startScript();