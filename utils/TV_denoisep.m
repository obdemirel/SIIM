function Z = TV_denoisep(U, aTV)



[m,n] = size(U);

aL1 = 0; 
iden = @(x) x;
opts = [];
opts.maxItr = 5; %400
opts.gamma = 1.0; %1.6
opts.relchg_tol = 5e-4;
opts.beta = 10; %5


 K2 = 1/sqrt(m*n)*fft2(U);
% K2 = 1/sqrt(m*n)*fft2c(U);
Z = RecPFp(m,n,aTV,aL1,1:m*n,K2(:),2,opts,iden,iden,max(abs(U(:))));
