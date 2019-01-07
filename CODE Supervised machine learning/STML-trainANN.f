cc ******************************************************************************************
cc
cc    ****************************************************
cc    *      COPYRIGHT: YI, ZHANG                        *
cc    *      LASSP, CORNELL UNIVERSITY                   *
cc    *      All Rights Reserved.    ----   2018. 02     *
cc    ****************************************************
cc
cc ******************************************************************************************

	implicit none
	cc	Parameter of the neural network. nn: number of neurons in the fully connected hidden layer;
	cc	ni: number of inputs; L: LxL is the 2-d system size; mini: mini-batch size; epoch: size of the epoch;
	cc	no: number of outputs; epoch: number of mini batches in an epoch;
	cc	ncount: the times eta is halved for more precise (slower) convergence;
	cc	nstop: the limit of ncount before the optimization terminates.
	integer nn, L, ni, no
	integer epoch, nepoch, ncount, nstop, nstep, mini
	parameter (nn=50, L=516, ni=L*L, no=4)
	parameter (mini=50, epoch=100, nstop=3, nstep=8)
	character*70 file1, file2, file3, file4
	character*70 file8, file5, file6, file7
cc	Hyper-parameters. eta: learning speed; lambda: L2 regulation; cost: cost function.
	double precision eta, lambda, cost
	parameter (lambda = 0.000001)
cc	The network values, weights and biases.
	double precision a1(nn), a2(no), delta1(nn), delta2(no), z1(nn)
	double precision w1(ni, nn), b1(nn), w2(nn, no), b2(no), z2(no)
	double precision dw1(ni, nn), db1(nn), dw2(nn, no), db2(no)
cc	The inputs. nval: size of validation data; nsap: size of training samples;
cc	accumax controls step size - no improvement in nstep -> smaller step.
cc	nfile: total number of data files.
	integer nval, nsap, accu, accustep, nfile, ifile
	parameter (nval=1000, nsap=9000, nfile=100)
	double precision accumax, pi, vx(ni,nsap+nval)
	integer cat(nsap+nval)
cc	vx and cat are the actual data sets.
cc	Running parameters. Info, work2 and temp are for matrix inversion.
	integer i, j, k, m, n, itemp, itemp2, info, io
	double precision d1, d2, d, temp
cc	Randam number parameters.
	integer*4 timearray(3)
cc	Initialize the random numbers.
	call itime(timearray)
	call SRAND(timearray(1)*3600+
     &	timearray(2)*60+timearray(3))
	pi = 4.0 * atan(1.0)
	eta = 0.01

cc	File to keep track of ANN #.
	open(19, file='filelog.dat')
	read(19, *), info
	close(19)

cc	The log file and parameter (weights and biases) file of the #th ANN.
	file3='traininglog_full_wd00nobia.dat'
	file4='network_full_wd00nobia.dat'
        file3(20:20) = char(ichar(file3(20:20))+info/10)
        file3(21:21) = char(ichar(file3(21:21))+info-info/10*10)
        file4(16:16) = char(ichar(file4(16:16))+info/10)
        file4(17:17) = char(ichar(file4(17:17))+info-info/10*10)

cc		Update ANN # file.
	open(19, file='filelog.dat')
	write(19, *), info+1
	close(19)

	do 3 i = 1, ni
	do 3 j = 1, nsap+nval
	vx(i, j) = 0.0
3	continue

cc	Loading training set.
	file1='data3/data25batch4cats4dislA1307_00.csv'
	do 5 ifile = 0, nfile-1
	file2 = file1
	file2(34:34) = char(ichar(file2(34:34))+ifile/10)
	file2(35:35) = char(ichar(file2(35:35))+ifile-ifile/10*10)
	open(20, file=file2)
	do 35 k = 1, 100
	read(20, *), vx(:, ifile*100+k), cat(ifile*100+k)
35	continue
	print*, 'File ', file2, ' processed!'
	close(20)
5	continue

cc	Overwrite with the new cat2 with 'no bias'.
	file5='data3/data100batch4cats4dislA1307_sameCat2only_00.csv'
	itemp = 0
	do 125 ifile = 1, 25
	file2 = file5
	file2(48:48) = char(ichar(file2(48:48))+ifile/10)
	file2(49:49) = char(ichar(file2(49:49))+ifile-ifile/10*10)
	open(20, file=file2)
	do 135 k = 1, 100
140	continue
	itemp = itemp + 1
	if(cat(itemp) .ne. 1) goto 140
	read(20, *), vx(:, itemp), itemp2
135	continue
	print*, 'File ', file2, ' processed!'
	close(20)
125	continue
	print*, 'Final cat2 = ', itemp
cc	All old cat2 data is overwritten.

cc	Initializing the neural network with Box-Muller transformation for Gaussian distribution.
cc	Bias (weight) initialization with standard deviation of 1 (1/sqrt(input)).
	do 10 i = 1, nn
15	d1 = 2.0 * rand() - 1.0
	d2 = 2.0 * rand() - 1.0
	d = d1*d1+d2*d2
	if(d .ge. 1.0) goto 15
	d = sqrt((-2.0 * log(d))/d)
	b1(i) = d1*d
	do 18 j = 1, no
	w2(i, j) = d2*d/sqrt(1.0*nn)
18	continue
	do 20 j = 1, ni/2
25	d1 = 2.0 * rand() - 1.0
	d2 = 2.0 * rand() - 1.0
	d = d1*d1+d2*d2
	if(d .ge. 1.0) goto 25
	d = sqrt((-2.0 * log(d))/d)
	w1(j*2-1, i) = d1*d/sqrt(1.0*ni)
	w1(j    , i) = d2*d/sqrt(1.0*ni)
20	continue
10	continue
30	d1 = 2.0 * rand() - 1.0
	d2 = 2.0 * rand() - 1.0
	d = d1*d1+d2*d2
	if(d .ge. 1.0) goto 30
	d = sqrt((-2.0 * log(d))/d)
	do 28 j = 1, no/2
	b2(j*2-1) = d1*d
	b2(j*2  ) = d2*d
28	continue
	print*, 'Initialization complete.'
	ncount = 0
	nepoch = 0
	accustep = 0
	accumax = 0.0
cc	Initialization complete.

cc	Learning loops.
200	nepoch = nepoch + 1
cc	A new epoch.
	cost = 0.0
	do 205 n = 1, epoch
cc	A new mini batch. Initialize the gradients for the weights and biases.
	do 210 i = 1, nn
	db1(i) = 0.0
	do 215 j = 1, no
	dw2(i, j) = 0.0
215	continue
	do 225 j = 1, ni
	dw1(j, i) = 0.0
225	continue
210	continue
	do 220 i = 1, no
	db2(i) = 0.0
220	continue
	do 275 m = 1, mini
cc	Pick a training sample.
	itemp = nsap* rand() + 1
	if(itemp .gt. nsap) itemp = itemp - 1
	info = cat(itemp) + 1
cc	Feed it forward.
	do 230 i = 1, no
	z2(i) = 0.0
230	continue
	do 240 k = 1, nn
	z1(k) = 0.0
	do 250 i = 1, ni
	z1(k) = z1(k) + vx(i, itemp) * w1(i, k)
250	continue
	z1(k) = z1(k) + b1(k)
	a1(k) = 1.0/(1.0+ exp(-z1(k)))
	do 245 j = 1, no
	z2(j) = z2(j) + w2(k, j) * a1(k)
245	continue
240	continue
	temp = 0.0
	do 255 i = 1, no
	z2(i) = z2(i) + b2(i)
	temp = temp + exp(z2(i))
255	continue
	do 258 i = 1, no
	a2(i) = exp(z2(i))/temp
cc	Calculate the errors.
 	delta2(i) = a2(i)
	db2(i) = db2(i) + delta2(i)
258	continue
	if(itemp .le. nsap) cost = cost - log(a2(info))
	delta2(info) = delta2(info) - 1.0
	db2(info) = db2(info) - 1.0
cc	Back propagation for the errors and gradients for the biases.
	do 260 i = 1, nn
	temp = 0.0
	do 265 j = 1, no
	temp = temp + delta2(j) * w2(i, j)
265	continue
	delta1(i) = temp * a1(i) * (1.0 - a1(i))
	db1(i) = db1(i) + delta1(i)
260	continue
cc	Calculate the gradient function for the weights.
	do 270 i = 1, nn
	do 273 j = 1, no
	dw2(i,j) = dw2(i,j) + delta2(j) * a1(i)
273	continue
	do 278 j = 1, ni
	dw1(j, i) = dw1(j, i) + delta1(i) * vx(j, itemp)
278	continue
270	continue
cc	Weights and biases gradients updated for one sample.
275	continue
cc	Update the new weights and biases for the current mini batch.
	do 280 i = 1, nn
	b1(i) = b1(i) - db1(i) * eta  / mini
	do 283 j = 1, no
	w2(i, j) = w2(i, j)*(1.0-lambda) - dw2(i, j) * eta / mini
283	continue
	do 288 j = 1, ni
	w1(j, i) = w1(j, i)*(1.0-lambda) - dw1(j, i) * eta / mini
288	continue
280	continue
	do 290 i = 1, no
	b2(i) = b2(i) - db2(i) * eta / mini
290	continue
cc	Current mini batch finished.
205	continue
cc	Epoch completed. Validation.
	accu = 0
	do 300 m=nsap+1, nsap+nval
	info = cat(m)+1
cc	The known category.
	do 305 i = 1, no
	z2(i) = 0.0
305	continue
	do 310 k = 1, nn
	z1(k) = 0.0
	do 315 i = 1, ni
	z1(k) = z1(k) + vx(i, m) * w1(i, k)
315	continue
	z1(k) = z1(k) + b1(k)
	a1(k) = 1.0/(1.0+ exp(-z1(k)))
	do 320 j = 1, no
	z2(j) = z2(j) + w2(k,j) * a1(k)
320	continue
310	continue
cc	Find the category with the max ANN output.
	temp = -100.0
	do 325 j = 1, no
	z2(j) = z2(j) + b2(j)
	if(z2(j) .gt. temp) then
	temp = z2(j)
	itemp = j
	endif
325	continue
cc Answer correct? -> Accuracy.
	if(itemp .eq. info) accu = accu + 1
300	continue
	print*, 'Epoch', nepoch, 'accuracy =', accu*1.0/nval
	print*, 'Slowdown #=', ncount, 'cost=', cost/epoch/mini
	open(22, file=file3)
	do
	read(22, *, iostat=io), itemp2, temp, d
	if(io .lt. 0) exit
	end do
	backspace(22)
	write(22, *), nepoch, accu*1.0/nval, cost/epoch/mini
	close(22)
	if (accu*1.0/nval .gt. accumax) then
	accumax = accu*1.0/nval
	accustep = 0
	else
	accustep = accustep + 1
	print*, 'No improvement in', accustep
	endif
cc	Self-termination and slow-down routine.
	if(accustep .gt. nstep .AND. accu*1.0/nval .gt. 0.96) then
	ncount = ncount + 1
	eta = eta / 2
	accumax = 0.0
	endif
	if(ncount .le. nstop) goto 200
	print*, 'Training complete.'

cc	Output ANN data.
	open(21, file=file4)
c	write(21,*), 'input', 'hidden', 'w1'
	do 360 i = 1, ni
	do 360 k = 1, nn
	write(21,*), i, k, w1(i, k)
360	continue
c	write(21,*), 'hidden', 'b1'
	do 365 k = 1, nn
	write(21,*), k, b1(k)
365	continue
c	write(21,*), 'hidden', 'w2'
	do 370 k = 1, nn
	do 370 i = 1, no
	write(21,*), k, w2(k, i)
370	continue
c	write(21,*), 'b2'
	do 375 i = 1, no
	write(21,*), b2(i)
375	continue
	close(21)

	end
