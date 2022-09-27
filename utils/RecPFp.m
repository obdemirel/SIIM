function [U,Out] = RecPFp(m,n,aTV,aL1,picks,B,TVtype,opts,PsiT,Psi,URange,uOrg)
% [U,Out] = RecPF(m,n,aTV,aL1,picks,B,TVtype,opts,PsiT,Psi,URange,uOrg)
%
% RecPF solves the TVL1-L2 model:
%
%   min aTV*TV(u) + aL1*|PsiT*U|_1 + 0.5|Fp*U - B|_2^2
%
% Inputs:
%
%  aTV, aL1 -- regularization parameters in the model (positive)
%           IMPORTANT: see Lines 60, 66, 67 for data/parameter normalization
%  picks    -- sample positions in Fourier domain
%  B        -- measurment vector
%  TVtype   -- 2 (isotropic) or 1 (anisotropic)
%  opts     -- parameters for algorithm
%  PsiT     -- sparsifying basis of U, 
%              x = PsiT*U is the sparse representation of U under Psi
%  Psi      -- inverse of PsiT, Psi*x = U reconstructs the image
%  URange   -- grayscale range of U, e.g., 1, 255, 65535
%  uOrg     -- (optional) true image
%
% Outputs: 
%          
%  U      -- reconstructed image
%  Out    -- iteration information, e.g., iter number, relative errors,
%            function values, etc.

%
% Yin Zhang, 11-10-2007 
% Junfeng Yang, modified on Oct. 29, 2008
% Wotao Yin, modified in Spring 2009; added Bregman
%
% CAAM, Rice University, Copyright (2009)

%% set options
    % --- sample setting for noiseless measurements --- 
%     maxItr = 50;        % max # of iterations
%     gamma = 1.6;        % noiseless choice = 1.6
%     beta = 8;           % noiseless choice = 200
%     relchg_tol = 1e-5;    
    % --- sample setting for noisy measurements --- 
%     maxItr = 100;       % max # of iterations
%     gamma = 1.0;        % noisy choice = 1.0
%     beta = 10;          % noisy choice = 10
%     relchg_tol = 5e-4;  % stopping tolerance based on relative change
    % --- getting from opts ---
    maxItr = opts.maxItr;
    gamma = opts.gamma;
    beta = opts.beta;
    relchg_tol = opts.relchg_tol;

bPrint  = false;    % screen display switch; turning it on slows the code down
bComplex = true;    % use complex computation or not

U = zeros(m,n);     % initial U. 
                    % If changing to anything nonzeor, you must change the 
                    % initialization of Ux and Uy below

%% normalize parameters and data
fctr = 1/URange;
B = fctr*B;
if exist('uOrg','var'); uOrg = fctr*uOrg; snr(U,uOrg); end

if (aTV <= 0); error('aTV must be strictly positive'); end
if (aL1 < 0); error('aL1 must be positive'); end
aTV = nnz(picks)/sqrt(m*n)*aTV;
aL1 = nnz(picks)/sqrt(m*n)*aL1;

%% initialize constant parts of numinator and denominator (in order to save computation time)
Numer1 = zeros(m,n); Numer1(picks) = sqrt(m*n)*B;
Denom1 = zeros(m,n); Denom1(picks) = 1;
prd = sqrt(aTV*beta);
Denom2 = abs(psf2otf([prd,-prd],[m,n])).^2 + abs(psf2otf([prd;-prd],[m,n])).^2;
if aL1 == 0; 
    Denom = Denom1 + Denom2;
else
    Denom = Denom1 + Denom2 + aL1*beta; 
end

%% initialize constants
Ux = zeros(m,n); Uy = zeros(m,n);
bx = zeros(m,n); by = zeros(m,n);
if (aL1 > 0);
    PsiTU = zeros(m,n); % equal to PsiT(U) as U=0; 
    z = zeros(m,n);
    d = zeros(m,n);
end

%% Main loop
for ii = 1:maxItr
        
    % ================================
    %  Begin Alternating Minimization
    % ----------------
    %   W-subprolem
    % ----------------
    switch TVtype
        case 1;   % anisotropic TV
            Ux = Ux + bx; Uy = Uy + by;      % latest Ux and Uy are already calculated
            Wx = sign(Ux).* max(abs(Ux)-1/beta,0);
            Wy = sign(Uy).* max(abs(Uy)-1/beta,0);
        case 2;   % isotropic TV
            UUx = Ux + bx; UUy = Uy + by;
            V = sqrt(abs(UUx).^2 + abs(UUy).^2);
            V = max(V - 1/beta, 0) ./ max(V,eps);
            Wx = V.*UUx; Wy = V.*UUy; 
        otherwise; 
            error('TVtype must be 1 or 2');
    end

    % ----------------
    %   Z-subprolem
    % ----------------
    if aL1 > 0;
        PsiTU = PsiTU + d;
        Z = sign(PsiTU).*max(abs(PsiTU)-1/beta,0);
    end;

    % ----------------
    %   U-subprolem
    % ----------------
    Uprev = U;
    
    rhs = (aTV*beta)*(DxtU(Wx-bx)+DytU(Wy-by)); %rhs = Compute_rhs_DxtU_DytU(Wx,Wy,bx,by,(aTV*beta)); % = (aTV*beta)*(DxtU(Wx-bx)+DytU(Wy-by));
    if (aL1 > 0); rhs = rhs + (aL1*beta)*Psi(Z-d); end

    U = ifft2((Numer1 + fft2(rhs))./Denom);
    
    if ~bComplex; U=real(U); end
    
    if aL1 > 0; PsiTU = PsiT(U); end
    Ux = cat(2, diff(U,1,2), U(:,1,:) - U(:,n,:));
    Uy = cat(1, diff(U,1,1), U(1,:,:) - U(m,:,:));
    
    %
    %  End Alternating Minimization
    % ================================

    % -------------------------------------------------
    % check stopping criterion
    %
    relchg = norm(U-Uprev,'fro')/norm(U,'fro');
    if bPrint; 
        fprintf('itr=%d relchg=%4.1e', ii, relchg);
        if exist('uOrg','var'); fprintf(' snr=%4.1f',snr(real(U))); end
        fprintf('\n');
    end

    if relchg < relchg_tol
        break;
    end
    
    % ------------------------------------------
    % Bregman update
    %
    bx = bx + gamma*(Ux - Wx);
    by = by + gamma*(Uy - Wy);
    if (aL1 > 0); d = d + gamma*(PsiTU - z); end

end % outer

Out.iter = ii;

%% reverse normalization
%U = real(U)/fctr;
U = U/fctr;

end