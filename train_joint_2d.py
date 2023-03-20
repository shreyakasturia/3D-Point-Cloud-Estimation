import numpy as np
import scipy.misc,scipy.io
import time,os,sys
import threading
# import util

print('********Phase 2 training - joint 2D optimization with novel viewpoints**********')

import tensorflow as tf
# import data,graph,transform
# import options

from misc import util
import dataloader as data
from misc import transform
from model import cnn as graph
from misc import options


tf = tf.compat.v1
tf.disable_eager_execution()

opt = options.set(training=True)

# create directories for model output
util.mkdir("models_{0}".format(opt.group))

print('Loading model...')
tf.reset_default_graph()
# build graph
with tf.device("/gpu:0"):
	# ------ define input data ------
	inputImage = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.inH,opt.inW,3])
	renderTrans = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.novelN,4])
	depthGT = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.novelN,opt.H,opt.W,1])
	maskGT = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.novelN,opt.H,opt.W,1])
	PH = [inputImage,renderTrans,depthGT,maskGT]
	# ------ build encoder-decoder ------
	encoder = graph.encoder if opt.arch=="original" else \
			  graph.encoder_resnet if opt.arch=="resnet" else None
	decoder = graph.decoder if opt.arch=="original" else \
			  graph.decoder_resnet if opt.arch=="resnet" else None
	latent = encoder(opt,inputImage)
	XYZ,maskLogit = decoder(opt,latent) # [B,H,W,3V],[B,H,W,V]
	mask = tf.to_float(maskLogit>0)
	# ------ build transformer ------
	fuseTrans = tf.nn.l2_normalize(opt.fuseTrans,dim=1)
	XYZid,ML = transform.fuse3D(opt,XYZ,maskLogit,fuseTrans) # [B,1,VHW]
	newDepth,newMaskLogit,collision = transform.render2D(opt,XYZid,ML,renderTrans) # [B,N,H,W,1]
	# ------ define loss ------
	loss_depth = graph.masked_l1_loss(newDepth-depthGT,tf.equal(collision,1))/(opt.batchSize*opt.novelN)
	loss_mask = graph.cross_entropy_loss(newMaskLogit,maskGT)/(opt.batchSize*opt.novelN)
	loss = loss_mask+opt.lambdaDepth*loss_depth
	# ------ optimizer ------
	lr_PH = tf.placeholder(tf.float32,shape=[])
	optim = tf.train.AdamOptimizer(learning_rate=lr_PH).minimize(loss)
	# ------ generate summaries ------
	summaryImage = [util.imageSummary(opt,"image_RGB",inputImage,opt.inH,opt.inW),
					util.imageSummary(opt,"image_depth/pred",((1-newDepth)*tf.to_float(tf.equal(collision,1)))[:,0,:,:,0:1],opt.H,opt.W),
					util.imageSummary(opt,"image_depth/GT",(1-depthGT)[:,0,:,:,0:1],opt.H,opt.W),
					util.imageSummary(opt,"image_mask/new",tf.sigmoid(newMaskLogit[:,0,:,:,0:1]),opt.H,opt.W),
					util.imageSummary(opt,"image_mask",tf.sigmoid(maskLogit[:,:,:,0:1]),opt.outH,opt.outW),
					util.imageSummary(opt,"image_mask/GT",maskGT[:,0,:,:,0:1],opt.H,opt.W)]
	summaryImage = tf.summary.merge(summaryImage)
	summaryLoss = [tf.summary.scalar("loss_total",loss),
				   tf.summary.scalar("loss_mask",loss_mask),
				   tf.summary.scalar("loss_depth",loss_depth)]
	summaryLoss = tf.summary.merge(summaryLoss)

# load data
print("loading dataset...")
dataloader = data.Loader(opt)
dataloader.loadChunk(opt)

# prepare model saver/summary writer
saver = tf.train.Saver()
summaryWriter = tf.summary.FileWriter("summary_{0}/{1}".format(opt.group,opt.model))

print("Training")
timeStart = time.time()
# start session
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True
with tf.Session(config=tfConfig) as sess:
	sess.run(tf.global_variables_initializer())
	if opt.fromIt!=0:
		util.restoreModelFromIt(opt,sess,saver,opt.fromIt)
	elif opt.load:
		util.restoreModel(opt,sess,saver)
		summaryWriter.add_graph(sess.graph)

	chunkResumeN,chunkMaxN = opt.fromIt//opt.itPerChunk,opt.toIt//opt.itPerChunk
	# training loop
	for c in range(chunkResumeN,chunkMaxN):
		dataloader.shipChunk()
		dataloader.thread = threading.Thread(target=dataloader.loadChunk,args=[opt])
		dataloader.thread.start()
		for i in range(c*opt.itPerChunk,(c+1)*opt.itPerChunk):
			lr = opt.lr*opt.lrDecay**(i//opt.lrStep)
			# make training batch
			batch = data.makeBatch(opt,dataloader,PH)
			batch[lr_PH] = lr
			# run one step
			runList = [optim,loss,loss_depth,loss_mask,maskLogit]
			_,l,ld,lm,ml = sess.run(runList,feed_dict=batch)
			if (i+1)%100==0:
				print("epoch. {0}/{1}, lr={2}, loss={4} ({5},{6}), time={3}"
					.format(util.toCyan("{0}".format((i+1)/100)),
							opt.toIt/100,
							util.toYellow("{0:.0e}".format(lr)),
							util.toGreen("{0:.2f}".format(time.time()-timeStart)),
							util.toRed("{0:.2f}".format(l)),
							util.toRed("{0:.2f}".format(ld)),
							util.toRed("{0:.2f}".format(lm))))
			if (i+1)%100==0:
				summaryWriter.add_summary(sess.run(summaryLoss,feed_dict=batch),(i+1)/100)
			if (i+1)%1000==0:
				summaryWriter.add_summary(sess.run(summaryImage,feed_dict=batch),(i+1)/100)
			if (i+1)%1000==0:
				util.saveModel(opt,sess,saver,(i+1)/100)
				print(util.toGreen("model saved: {0}/{1}, it.{2}".format(opt.group,opt.model,(i+1)/100)))
		dataloader.thread.join()

print(util.toYellow("======= TRAINING DONE ======="))
