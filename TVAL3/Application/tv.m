%% Pragya Sharma

%% Clear, add path
clear; 
path(path,genpath(pwd));

%% Set up problem
load ('V17.mat'); % See Info. 
load ('e17.mat');
[m,n] = size(e);

%% Run TVAL
opts.mu = 2^8;
opts.beta = 2^5;
opts.mu0 = 2^1;
opts.beta0 = 2^1;
opts.tol = 1E-3;
opts.maxit = 400;
% opts.TVnorm = 1; % For anisotropic
opts.nonneg = true; % x needs to be positive
opts.disp = true;
opts.TVL2 = true;
opts.init = 0;

t = cputime;
[x, out] = TVAL3(e,V,n,1,opts);
t = cputime - t;
display(t);

a = sqrt(n);
x = reshape(x,[a,a]);
figure
mesh(abs(x)')
axis 'tight'
axis 'square'
