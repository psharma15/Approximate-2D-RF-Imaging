%% Code for generating RF image of a object in an experimental setup
% Based on paper 'Sparse passive radar imaging based on digital video
% broadcasting satelliters using music algorithm

function [] = main_od_music
clear all;

%% Generating forward problem
% Defining all variables here
c = physconst('LightSpeed');
f = 'f3';
Ntag = 4; 
Nrecv = 12;
object = 'var';
SNR = 20;
l = 1; % Defining l; Scaling factor for the image area
b = 1; % Defining scaling for receivers
m = 1; % Choosing scaling factor of scatterer position
k1 = 0; % Choosing shifting factor of scatterer pos in x and y dir
k2 = -0.0;
RCS=1;

% position of tags, receivers and object scatterers
switch(f)
    case 'f3'
        Freq =(1.7:0.05:2.2)*1e9;
    case 'f5'
        Freq = [1.79 2.03 2.2]*1e9; 
    case 'f6'
        Freq = 2e9;
    otherwise
        display('Error in frequency');
end
Nfreq = length(Freq);
assignin('base','Freq',Freq);
Pos_tag = zeros(Ntag,2);
switch(Ntag)
    case 32
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
    otherwise
        disp('Error in Pos_tag');
end
Pos_tag = l* Pos_tag;
Pos_recv = zeros(Nrecv,2);
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
    case 12
        Pos_recv = zeros(Nrecv,2);
        for i=1:Nrecv/3
            Pos_recv(i, :)=[-0.3+(i-1)*0.2 -0.5] ;
        end
        for i=1:Nrecv/3
            Pos_recv(i+(Nrecv/3),:)=[0.5 -0.3+(i-1)*0.2] ;
        end
        for i=1:Nrecv/3
            Pos_recv(i+(2*Nrecv/3), :)=[-0.5 -0.3+(i-1)*0.2] ;
        end   
    otherwise
        disp('Error in Pos_recv');
end
Pos_recv = b* Pos_recv;
figure;
plot(Pos_tag(:,1),Pos_tag(:,2),'ro');
hold on;
plot(Pos_recv(:,1),Pos_recv(:,2),'bo','MarkerFaceColor','b','MarkerSize',8);
switch(object)
    case 'sparse'
        Pos_s = [0.0, 0; 
                0.02, 0.25;
                -0.1, 0.06;
                0.18,0.12;
                -0.28,-0.25;
                0.3,-0.1;
                -0.08,-0.3;
                ]; 
    case 'sparse2'
                Pos_s = [0.0, -0.02; 
                0.0, 0.02;
                -0.02, 0.0;
                0.28,0.12;
                0.25,0.12;
                0.265,0.09;        
                ];             
     case 'circlefill'
        Pos_s=[];
        i=0;
        dtheta=1.2*pi/8;
        
        a0 = 0.05; 
        for a = 0.01:0.02:a0
            theta=0;
            while theta<=2*pi-dtheta/2
                i=i+1;
                Pos_s(i,:)=(a.*[cos(theta) sin(theta)]);
                theta=theta+dtheta;
            end
            i=i+1;
            dtheta=dtheta-pi/64;
        end
        ro=a*m*100;
        assignin('base','ro',ro);
    otherwise 
        disp('Error in Pos_s');
end
Pos_s = m * Pos_s;
[m1,~]=size(Pos_s);
Pos_s = Pos_s - k1* [zeros(m1,1), ones(m1,1)];
Pos_s = Pos_s - k2* [ones(m1,1), zeros(m1,1)];
plot(Pos_s(:,1),Pos_s(:,2),'k*');
axis 'square'
axis 'tight'
legend('Pos tag','Pos recv','Pos Scatterer'); 
% Compute phi_tag and phi_recv
x_sc=mean(Pos_s(:,1));
y_sc=mean(Pos_s(:,2));
phi_tag=zeros(1,Ntag);
for i=1:Ntag
    x_d=Pos_tag(i,1)-x_sc;
    y_d=Pos_tag(i,2)-y_sc;
    phi_tag(1,i)=angle(x_d+sqrt(-1)*y_d);
end
phi_recv=zeros(1,Nrecv);
for i=1:Nrecv
    x_d=Pos_recv(i,1)-x_sc;
    y_d=Pos_recv(i,2)-y_sc;
    phi_recv(1,i)= angle(x_d+sqrt(-1)*y_d);
end

% This is K = (Kx,Ky). 
Kx=zeros(Ntag*Nrecv*Nfreq);
Ky=zeros(Ntag*Nrecv*Nfreq);
count = 1;
for i=1:length(Freq)
    for m=1:length(phi_tag)
        for n=1:length(phi_recv)
            Kx(count) = Freq(i)/c*(cos(phi_tag(m))+cos(phi_recv(n)));
            Ky(count) = Freq(i)/c*(sin(phi_tag(m))+sin(phi_recv(n)));
            count = count+1;
        end
    end
end
figure;
plot(Kx,Ky,'r+');
hold on;
theta = 0:0.01:2*pi;
x = 2*max(Freq)/c*cos(theta);
y = 2*max(Freq)/c*sin(theta);
plot(x, y)
axis square

% Generate Rician fading factors
An=sqrt(1/2*10^(-SNR/10)); 
[k1, ~]=size(Pos_tag);
[k2, ~]=size(Pos_recv);
Rician_Factor=zeros(k1,k2,length(Freq));
Rician_Factor_Phase=zeros(length(Freq)*k1*k2,1);
Rician_Factor_Amp=zeros(length(Freq)*k1*k2,1);
count =1;
for i=1:length(Freq)
    for m=1:k1
        for n=1:k2
            X=1+An*randn(1,1);  % is 1 for LOS? (Rician Fading)
            Y=An*randn(1,1);
            Z=(X+sqrt(-1)*Y);
            Rician_Factor_Phase(count)= angle(Z);
            Rician_Factor_Amp(count)= abs(Z);
            Rician_Factor(m,n,i)=Z;
            count=count+1;
        end
    end
end

% generate signal for w/o and w/ scatterers case
G_wo_s=generation_signal_wo_scatter(Freq,Pos_tag,Pos_recv,Rician_Factor);
G_w_s=generation_signal_w_scatter(Freq,Pos_tag,Pos_recv,Pos_s,RCS,Rician_Factor);

% generate signal after calibration 
[k1, ~]=size(Pos_tag);
[k2, ~]=size(Pos_recv);
G_calib=zeros(k1,k2,length(Freq));
for i=1:length(Freq)
    freq=Freq(i);
    lamda=c/freq;
    for n=1:k2
        for m=1:k1
            p_tag=Pos_tag(m,:);
            p_recv=Pos_recv(n,:);
            r=norm(p_tag-p_recv);
            for n_pair=1:k2
                if n_pair~=n
                    G_w_s_ratio=G_w_s(m,n,i)/G_w_s(m,n_pair,i);
                    G_wo_s_ratio=G_wo_s(m,n,i)/G_wo_s(m,n_pair,i);
                    G_calib(m,n,i)=G_calib(m,n,i)+(G_w_s_ratio/G_wo_s_ratio-1)*exp(-1j*2*pi/lamda*r);
                end
            end
        end
    end
end
assignin('base','G_calib',G_calib);
% [all: freq, recv, tx/tags]
G = reshape(permute(G_calib,[3,2,1]),[],1);
assignin('base','G',G);

%% Recovery Section
J = fliplr(eye(Nrecv*Ntag*Nfreq)); 
R = G*G'; %  + J*(conj(G*G'))*J
assignin('base','R',R);
pause

[V,D] = eig(R);
[D,I] = sort(diag(D),'descend');
V = V(:, I);

O = V(:,1);
% O = fliplr(O);
x_v=(-.5:0.01: .5);
y_v=x_v;
p=combvec(x_v,y_v)';
lx=length(x_v);
P_music = zeros(lx*lx,1);
Freq = repmat(Freq',Ntag*Nrecv,1);
phi_tag0 = kron(atan(Pos_tag(:,1)./Pos_tag(:,2)), ones(Nrecv,1));
phi_recv0 = repmat(atan(Pos_recv(:,1)./Pos_recv(:,2)), Ntag,1);
for i = 1:lx*lx
    phi_scat = atan(p(i,1)/p(i,2));
    phi_tag = phi_tag0 ; % phi is x/y
    phi_recv = phi_recv0 ;
    Kx_mnf = (1/c)* Freq .* kron(sin(phi_tag) + sin(phi_recv), ones(Nfreq,1));
    Ky_mnf = (1/c)* Freq .* kron(cos(phi_tag) + cos(phi_recv), ones(Nfreq,1));
    const = p(i,1).*Kx_mnf + p(i,2) .*Ky_mnf;
    a = exp(1j * (2*pi) .* const);
    P_music(i) = (a'*a)/(a'*(O*O')*a);
    
end
        
image_output=reshape(P_music,[lx,lx]);

% Getting Object
C= (abs(image_output).^2)';
cmax = max(C(:)); %Making range 0-1, easier for thresholding
cmin = min(C(:));
C =(C-cmin)/(cmax-cmin); 
assignin('base','image_output',image_output);
assignin('base','C',C);
figure;
mesh(C);
axis 'square'; axis 'tight'
end

% generating signal without scatterer
function [G] = generation_signal_wo_scatter(Freq_v,Pos_tag,Pos_recv,Rician_Factor)

c=3e8;
[k1, ~]=size(Pos_tag);
[k2, ~]=size(Pos_recv);
G=zeros(k1,k2,length(Freq_v));
for i=1:length(Freq_v)
    for m=1:k1
        for n=1:k2
            freq=Freq_v(i);
            lamda=c/freq;
            p_tag=Pos_tag(m,:);
            p_recv=Pos_recv(n,:);
            r=norm(p_tag-p_recv);
            
            G(m,n,i)=Rician_Factor(m,n,i)*lamda/(4*pi*r)*exp(-1j*2*pi/lamda*r);
            if r == 0
                G(m,n,i) = 1e-4+1j*1e-4;
                
            end
        end
    end
end
end

% generating signal with scatterer
function [G] = generation_signal_w_scatter(Freq_v,Pos_tag,Pos_recv,Pos_s,RCS,Rician_Factor)
c=3e8;
[k1, ~]=size(Pos_tag);
[k2, ~]=size(Pos_recv);
[k3, ~]=size(Pos_s);
G=zeros(k1,k2,length(Freq_v));
for i=1:length(Freq_v)
    for m=1:k1
        for n=1:k2
            freq=Freq_v(i);
            lamda=c/freq;
            p_tag=Pos_tag(m,:);
            p_recv=Pos_recv(n,:);
            r=norm(p_tag-p_recv);
            
            G(m,n,i)=Rician_Factor(m,n,i)*lamda/(4*pi*r)*exp(-1j*2*pi/lamda*r);
            if r ==0
                G(m,n,i) = 1e-4+1j*1e-4;
            end
            for u=1:k3
                p_s=Pos_s(u,:);
                r1=norm(p_tag-p_s);
                r2=norm(p_s-p_recv);
                G(m,n,i)=G(m,n,i)+sqrt(RCS/4/pi/r1^2)*lamda/(4*pi*r2)*exp(-1j*2*pi/lamda*(r1+r2));
            end
        end
    end
end
end






