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
cc	nnn: the total number of neural networks included in the study.
	integer nn, L, ni, no, nnn, inn
	integer epoch, nepoch, ncount, nstop, nstep, mini
	parameter (nn=50, L=516, ni=L*L, no=4, nnn=82)
	parameter (mini=50, epoch=100, nstop=3, nstep=10)
	character*60 file1, file2, file3, file4, file5
cc	Hyper-parameters. eta: learning speed; lambda: L2 regulation; cost: cost function.
	double precision eta, lambda, cost
	parameter (lambda = 0.000001)
cc	The network values, weights and biases.
	double precision a1(nn), a2(no), delta1(nn), delta2(no), z1(nn)
	double precision w1(ni, nn), b1(nn), w2(nn, no), b2(no), z2(no)
	double precision dw1(ni, nn), db1(nn), dw2(nn, no), db2(no)
cc	The inputs. nval: size of validation data; nsap: size of training samples;
cc	accumax controls step size - no improvement in nstep -> smaller step.
	integer nval, nsap, accu, accustep, ifile
	parameter (nval=78, nsap=0)
	double precision accumax, pi, vx(ni,nsap+nval)
	double precision nnoutput(no, nnn, nval)
cc	Neural output: output neuron #, ANN #, sample #.
cc	Running parameters. Info, work2 and temp are for matrix inversion.
	integer i, j, k, m, n, itemp, itemp2, info
	double precision d, temp
	pi = 4.0 * atan(1.0)
	eta = 0.01

cc	Target file for network output.
	file1='new/network_full_wd00nobia.dat'

	cc	20 samples of doping dependence.
	file3='fix/dope00.csv'
	do 5 ifile = 1, 20
	file2 = file3
	file2(9:9) = char(ichar(file2(9:9))+ifile/10)
	file2(10:10) = char(ichar(file2(10:10))+ifile-ifile/10*10)
	open(20, file=file2)
	read(20, *), vx(:, ifile)
	print*, 'File ', file2, ' processed!'
	close(20)
5	continue

cc	New doping dependence data.
	file2='data3/OD62KnoDWjustCu6pixFC_zmap_516pix_crop1.csv'
	open(20, file=file2)
	read(20, *), vx(:, 21)
	print*, 'File ', file2, ' processed!'
	close(20)
	file2='data3/OD62KnoDWjustCu6pixFCR90S_zmap_516pix_crop1_rot90.csv'
	open(20, file=file2)
	read(20, *), vx(:, 22)
	print*, 'File ', file2, ' processed!'
	close(20)
	file2='data3/OD62KnoDWjustCu6pixFC_zmap_516pix_crop2.csv'
	open(20, file=file2)
	read(20, *), vx(:, 23)
	print*, 'File ', file2, ' processed!'
	close(20)
	file2='data3/OD62KnoDWjustCu6pixFC_zmap_516pix_crop2_rot90.csv'
	open(20, file=file2)
	read(20, *), vx(:, 24)
	print*, 'File ', file2, ' processed!'
	close(20)
	file2='data3/OD62KnoDWjustCu6pixFC_zmap_516pix_crop3.csv'
	open(20, file=file2)
	read(20, *), vx(:, 25)
	print*, 'File ', file2, ' processed!'
	close(20)
	file2='data3/OD62KnoDWjustCu6pixFC_zmap_516pix_crop3_rot90.csv'
	open(20, file=file2)
	read(20, *), vx(:, 26)
	print*, 'File ', file2, ' processed!'
	close(20)
	file2='data3/OD62KnoDWjustCu6pixFC_zmap_516pix_crop4.csv'
	open(20, file=file2)
	read(20, *), vx(:, 27)
	print*, 'File ', file2, ' processed!'
	close(20)
	file2='data3/OD62KnoDWjustCu6pixFC_zmap_516pix_crop4_rot90.csv'
	open(20, file=file2)
	read(20, *), vx(:, 28)
	print*, 'File ', file2, ' processed!'
	close(20)

	file4='fix/UD45KZrE000_6pixFC_zmap_516pix.csv'
	file5='fix/UD45KZrE000_6pixFC_zmap_516pix_rot90.csv'
cc	25 energy dependenc samples for each direction.
	do 25 ifile = 1, 25
	file2 = file4
	itemp = ifile * 6
        file2(13:13) = char(ichar(file2(13:13))+itemp/100)
	file2(14:14) = char(ichar(file2(14:14))+(itemp-itemp/100*100)/10)
	file2(15:15) = char(ichar(file2(15:15))+itemp-itemp/10*10)
	open(20, file=file2)
	read(20, *), vx(:, ifile+28)
	print*, 'File ', file2, ' processed!'
	close(20)
25	continue
	do 45 ifile = 1, 25
	file2 = file5
	itemp = ifile * 6
        file2(13:13) = char(ichar(file2(13:13))+itemp/100)
	file2(14:14) = char(ichar(file2(14:14))+(itemp-itemp/100*100)/10)
	file2(15:15) = char(ichar(file2(15:15))+itemp-itemp/10*10)
	open(20, file=file2)
	read(20, *), vx(:, ifile+28+25)
	print*, 'File ', file2, ' processed!'
	close(20)
45	continue

cc	File for the neural outputs.
	open(24, file = 'nnoutputs.dat')

	do 105 inn = 1, nnn
cc	Load the inn'th neural network.
	file5 = file1
        file5(20:20) = char(ichar(file5(20:20))+inn/10)
        file5(21:21) = char(ichar(file5(21:21))+inn-inn/10*10)

	open(21, file=file5)
	do 360 i = 1, ni
	do 360 k = 1, nn
	read(21,*), itemp, itemp2, w1(i, k)
360	continue
	do 365 k = 1, nn
	read(21,*), itemp2, b1(k)
365	continue
	do 370 k = 1, nn
	do 370 i = 1, no
	read(21,*), itemp2, w2(k, i)
370	continue
	do 375 i = 1, no
	read(21,*), b2(i)
375	continue
	close(21)

	print*, 'Network ', file5, ' processed!'
cc	Loading complete. Now obtain the neural output for the nval test samples.

	do 300 m=1, nval
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
	temp = -100.0
	d = 0.0
	do 325 j = 1, no
	z2(j) = z2(j) + b2(j)
	d = d + exp(z2(j))
	if(z2(j) .gt. temp) then
	temp = z2(j)
	itemp = j
	endif
325	continue
	do 328 i = 1, no
	a2(i) = exp(z2(i))/d
328	continue
	write(24, *), m, a2
	nnoutput(:, inn, m) = a2
300	continue

105	continue

cc	Analyze the statistics of the neural outputs.
	open(23, file='nnaverage.dat')

	do 505 itemp = 1, no
	do 505 m = 1, nval
cc	Average.
	temp = 0.0
	do 510 inn = 1, nnn
	temp = temp + nnoutput(itemp, inn, m)
510	continue
	temp = temp / nnn
cc	Error of the mean.
	d = 0.0
	do 520 inn = 1, nnn
	d = d +(nnoutput(itemp, inn, m)-temp) ** 2
520	continue
	d = sqrt(d/nnn/(nnn-1))
	write(23, *), m, temp, d
cc	Sample #, average, error of mean.
505	continue

	close(23)
	close(24)

	end
