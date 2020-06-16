const tfjs = require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');
const canvas = require('canvas');
const fs = require('fs');
const path = require('path');

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


const isDirectory = fpath => fs.statSync(fpath).isDirectory();
const getDirectories = fpath =>
    fs.readdirSync(fpath).map(name => path.join(fpath, name)).filter(isDirectory);

const isFile = fpath => fs.statSync(fpath).isFile();  
const getFiles = fpath =>
    fs.readdirSync(fpath).map(name => path.join(fpath, name)).filter(isFile);

const getFilesRecursively = (path) => {
    let dirs = getDirectories(path);
    let files = dirs
        .map(dir => getFilesRecursively(dir)) // go through each directory
        .reduce((a,b) => a.concat(b), []);    // map returns a 2d array (array of file arrays) so flatten
    return files.concat(getFiles(path));
};

async function startScript() {

  const files = getFilesRecursively('data');

  console.log(`Will load ${files.length} files !`);


  const fd = fs.openSync('./output.json', 'a');

  fs.writeSync(fd, '{');

  for(let i = 0; i < files.length; i++) {
    
    if(i % 1000 === 1) {
      console.log(`Loaded ${i}/${files.length} files`);
    }
    
    const img = await canvas.loadImage(files[i]);
    ctx.drawImage(img, 0, 0, WIDTH, HEIGHT);
    net = await getNet();
    const pose = await net.estimateSinglePose(cv);

    if(i != files.length - 1) {
      fs.writeSync(fd, `"${files[i]}"` + ':' + JSON.stringify(pose) + ",\n");
    } else {
      fs.writeSync(fd, `"${files[i]}"` + ':' + JSON.stringify(pose) + "}\n");
    }

  }
  console.log("Finished reading files !");

  fs.closeSync(fd); 

}

startScript();