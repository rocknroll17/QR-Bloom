import * as tf from '@tensorflow/tfjs-node';
// compare manual-pad+valid vs numeric pad for conv3d, k3s1p1 and k4s2p1
const x=tf.randomNormal([1,8,8,8,3]);
const w3=tf.randomNormal([3,3,3,3,5]);   // k3, in3 out5
const w4=tf.randomNormal([4,4,4,3,5]);   // k4
function manual(x,w,s,p){let h=p>0?tf.pad(x,[[0,0],[p,p],[p,p],[p,p],[0,0]]):x;return tf.conv3d(h,w,[s,s,s],'valid');}
for(const [w,s,p,name] of [[w3,1,1,'k3s1p1'],[w4,2,1,'k4s2p1']]){
  const a=manual(x,w,s,p);
  let b; try{ b=tf.conv3d(x,w,[s,s,s],p);}catch(e){console.log(name,'numeric pad threw:',e.message);continue;}
  const diff=tf.max(tf.abs(tf.sub(a,b))).arraySync();
  console.log(`${name}: shapes ${a.shape} vs ${b.shape}  max|diff|=${diff.toExponential(2)}`);
}
