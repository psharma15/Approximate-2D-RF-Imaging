%% Pragya Sharma

%% Clear, add path
clear; 
path(path,genpath(pwd));

%% Set up problem
load ('V23.mat'); % See Info. 
load ('e23.mat');
[~,n] = size(e);

%% Run TVAL
opts.nonneg = true; % x needs to be positive
opts.mu = 2^10;
opts.beta = 2^7;
opts.mu0 = 2^5;
opts.beta0 = 2^5;
opts.tol = 10^-(2.5); % Usual 1e-2 1e-3
opts.maxit = 400;
% opts.TVnorm = 1; % For anisotropic
opts.disp = true;
opts.TVL2 = true;
opts.init = 0;

t = cputime;
[x, out] = TVAL3(e,V,n,1,opts);
t = cputime - t;
display(t);

a = sqrt(n);
im = reshape(x,[a,a]);
figure
mesh(abs(im)')
axis 'tight'
axis 'square'
view(0,90)