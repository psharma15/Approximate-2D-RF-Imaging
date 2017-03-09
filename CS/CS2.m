%% Compressed Sensing solution for Indoor RF-imaging using l1-magic/ NESTA
% Pragya Sharma, March 02 2017 - l1-magic Working 

function CS2()
%% Clear, add path
clear; 
path1 = cd;                         % Take care to add from right folder
addpath([path1,'/Optimization']);
path2 = [path1,'/Data/11'];
addpath(path2);

%% Set up problem
load ('V11.mat'); 
load ('e11.mat');
load ('e_bwd11.mat');
[~,n] = size(e);

%% Run l1 - minimization
algo = 'l1magic';
switch algo
    case 'l1magic'
%         x0 = e'*V;              % Initial guess
        x0 = e_bwd'*V;              % Initial guess
        xp = l1eq_pd(x0, e, [], V);
    case 'NESTA'
        Sigma = 0.1;                % Noise level
        mu = 0.1*Sigma;             % Can be chosen to be small
        delta = 1e-3;
        opts = [];
        [xp,niter_2,~,~] = NESTA(e,[],V,mu,delta,opts);
        assignin('base','niter_2',niter_2);
    otherwise
        fprintf('Wrong algorithm chosen\n');
end

a = sqrt(n);
im = reshape(xp,[a,a]);
assignin('base','im',im);
figure
mesh((abs(im).^2)')
axis 'tight'
axis 'square'
axis([-10 90 -10 90])
xlabel('x-axis','FontSize',14)
ylabel('y-axis','FontSize',14)
view(0,90); colormap('cool');

rmpath(path2);
end