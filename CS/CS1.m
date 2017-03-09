%% Using l1-magic algorithm on previous (w/o RCS) forward model. Also,
% receiver signals are generated from this forward model.
% Option to choose Old or New (Improved) forward model.

%% Add path
path = cd;
addpath([path,'/Optimization']);
addpath([path,'/Data']);

%% Generate A matrix
c = physconst('LightSpeed');
Ntag = 16; 
Nrecv = 4;
f = 'f5'; % Choose frequency range to select random frequencies from
flag_rand = 0; % 1 == random only for frequency
fwdmod = 'new'; % ForWarD MODel: new, closer to real. Old, approximate.
obj = 'sparse'; % block or sparse
rng(3,'twister');
RCS = 0.1;
switch(f)     
    case 'f3'
        Freq =(1.7:0.005:2.2)*1e9;     
    case 'f5'
        Freq = [1.79 2.03 2.2]*1e9; 
    case 'f6'
        Freq = 2e9;
    otherwise
        display('Error in frequency');
end
if (flag_rand)
    Nfreq_temp = length(Freq);
    Nfreq = 11;
    q = randperm(Nfreq_temp);
    Freq = Freq(q(1:Nfreq));
else
    Nfreq = length(Freq);
end
switch(Ntag)
    case 32
        Pos_tag = zeros(Ntag,2);
        for i=1:Ntag/4
            Pos_tag(i, :)=[-0.35+(i-1)*0.1 -0.5] ;
        end
        for i=1:Ntag/4
            Pos_tag(i+(Ntag/4),:)=[0.5 -0.35+(i-1)*0.1] ;
        end
        for i=1:Ntag/4
            Pos_tag(i+(2*Ntag/4), :)=[-0.35+(i-1)*0.1 0.5] ;
        end
        for i=1:Ntag/4
            Pos_tag(i+(3*Ntag/4), :)=[-0.5 -0.35+(i-1)*0.1] ;
        end
    case 16   
        Pos_tag = zeros(Ntag,2);
        for i=1:Ntag/4
            Pos_tag(i, :)=[-0.3+(i-1)*0.2 -0.5] ;
        end
        for i=1:Ntag/4
            Pos_tag(i+(Ntag/4),:)=[0.5 -0.3+(i-1)*0.2] ;
        end
        for i=1:Ntag/4
            Pos_tag(i+(2*Ntag/4), :)=[-0.3+(i-1)*0.2 0.5] ;
        end
        for i=1:Ntag/4
            Pos_tag(i+(3*Ntag/4), :)=[-0.5 -0.3+(i-1)*0.2] ;
        end
    case 4
        Pos_tag = [-0.5,0.5;-0.1667,0.5;0.1667,0.5;0.5,0.5];
    case 'rand'
        Ntag = 16;
        Pos_tag = [(0.5-(-0.5)).*rand(Ntag/4,1) - 0.5, - 0.5*ones(Ntag/4,1);
            0.5*ones(Ntag/4,1), (0.5-(-0.5)).*rand(Ntag/4,1) - 0.5;
            (0.5-(-0.5)).*rand(Ntag/4,1) - 0.5, 0.5*ones(Ntag/4,1);
            -0.5*ones(Ntag/4,1), (0.5-(-0.5)).*rand(Ntag/4,1) - 0.5];
    otherwise
        disp('Error in Pos_tag');
end
switch(Nrecv)
    case 1
        Pos_recv = [-0.5,-0.5];
    case 2
        Pos_recv = [-0.5,-0.5;0.5,-0.5];   
    case 4
        Pos_recv = [-0.5,-0.5;0.5,-0.5;0.5,0.5;-0.5,0.5];
    case 6
        Pos_recv(:,1) = linspace(-0.5,0.5,6);
        Pos_recv(:,2) = ones(Nrecv,1).*-0.5;
    case 16
        Pos_recv = zeros(Nrecv,2);
        for i=1:Nrecv/4
            Pos_recv(i, :)=[-0.3+(i-1)*0.2 -0.5] ;
        end
        for i=1:Nrecv/4
            Pos_recv(i+(Nrecv/4),:)=[0.5 -0.3+(i-1)*0.2] ;
        end
        for i=1:Nrecv/4
            Pos_recv(i+(2*Nrecv/4), :)=[-0.3+(i-1)*0.2 0.5] ;
        end
        for i=1:Nrecv/4
            Pos_recv(i+(3*Nrecv/4), :)=[-0.5 -0.3+(i-1)*0.2] ;
        end
    case 'rand'
        Nrecv = 4;
        Pos_recv = [(0.5-(-0.5)).*rand(Nrecv/4,1) - 0.5, - 0.5*ones(Nrecv/4,1);
            0.5*ones(Nrecv/4,1), (0.5-(-0.5)).*rand(Nrecv/4,1) - 0.5;
            (0.5-(-0.5)).*rand(Nrecv/4,1) - 0.5, 0.5*ones(Nrecv/4,1);
            -0.5*ones(Nrecv/4,1), (0.5-(-0.5)).*rand(Nrecv/4,1) - 0.5];
    otherwise
        disp('Error in Pos_recv');
end

figure
plot(Pos_tag(:,1),Pos_tag(:,2),'ro');
hold on;
plot(Pos_recv(:,1),Pos_recv(:,2),'bo','MarkerFaceColor','b','MarkerSize',8);
axis 'square'

Ngrid = 80;
x_v = linspace(-0.4,0.4,Ngrid);
y_v = x_v;
lx = length(x_v);
P = combvec(x_v,y_v)';

R1 = repmat(sqrt(sum(abs( repmat(permute(P, [1 3 2]), [1 Ntag 1]) ...
- repmat(permute(Pos_tag, [3 1 2]), [lx*lx 1 1]) ).^2, 3)),1,Nrecv);

dummy1 = ones(1,Ntag);
R2 = kron(sqrt(sum(abs( repmat(permute(P, [1 3 2]), [1 Nrecv 1]) ...
- repmat(permute(Pos_recv, [3 1 2]), [lx*lx 1 1]) ).^2, 3)), dummy1);

R1 = repmat(R1,1,Nfreq); % All tag, recv, freq
R2 = repmat(R2,1,Nfreq); % All tag, recv, freq
R = R1 + R2; 
dummy3 = ones(1,Ntag*Nrecv); 
Freq = repmat(kron(Freq,dummy3),[lx*lx 1]);

switch fwdmod
    case 'old'
        % Previous forward model
        e = exp(-1j*(2*pi/c)*(Freq.*R));
        e = transpose(e);
    case 'new'
        % Improved forward model
        e1 = exp(-1j*(2*pi/c)*(Freq.*(R1+R2)));
        e_fwd = sqrt(RCS/(4*pi))*(1/(4*pi))*(c./(Freq.*R1.*R2)).*e1; 
        e_fwd = transpose(e_fwd); % == A matrix
        e = e_fwd;
    otherwise
        display ('Wrong option of Forward Model');
end
% Clear variables
clearvars -except D lx x_v y_v Freq Ngrid obj e1 e

%% Generating test input, sparse in spatial domain 
% Using identity representation basis(Can use gradient/TV instead) 
img = zeros(lx,lx);
switch obj
    case 'sparse'
        Nscat = 6;
        grid_index = 1:Ngrid;
        temp2 = randperm(Ngrid);
        xpos = grid_index(temp2(1:Nscat));
        temp3 = randperm(Ngrid);
        ypos = grid_index(temp3(1:Nscat));
        for i = 1:Nscat
            img(xpos(i),ypos(i)) = 1;
        end
    case 'block'
%         img(39:42,17:23) = 1;
        img(49:52,17:22) = 1;
    otherwise
        fprintf('This object doesn''t exist.\n');
end
[X, Y] = meshgrid(x_v,y_v);
figure
mesh(X,Y,img'); colormap('gray'); axis 'square'; view(0,90)
x = reshape(img,[],1);

% Fourier reconstruction
std_dev = 0.00; % Should be scaled wrt to e*x
e_bwd = transpose(e1);
[l,~] = size(e);
V = e*x ;
rand_noise = std_dev*randn(l,1).*V;
V = V + rand_noise;
xFourier = e_bwd'*V; xFourier = abs(xFourier);
xFourier = (reshape(xFourier, [lx lx]));
figure
mesh(X,Y,xFourier')
view(0,90); axis 'square'; axis 'tight' 

%% Perform compressed sensing using l1-magic
% Results worse than Fourier/ TV for synthetic data for bloack object.
% Perfect results for sparse object - 1 pixel, better than TVAL3, new
% fwdmod
x0 = e.'*V;
xp = l1eq_pd(x0, e, [], V);
xCS = reshape(xp,lx,lx);
xCS = abs(xCS); % Sensitive to noise: rand_noise
figure
mesh(X,Y,xCS')
view(0,90); colormap('cool'); axis 'square'; axis 'tight' 
