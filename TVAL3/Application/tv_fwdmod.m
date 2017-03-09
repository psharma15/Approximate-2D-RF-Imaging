%% This is using forward model for maxent by 17th Feb 2017
% Signal is generated based on Rician Fading model, point scatterers.
% Last edit: 21-02-17
% Pragya Sharma
function tv_fwdmod()
    c = physconst('LightSpeed');
    f = 'f3';
    Ntag = 16; 
    Nrecv = 4;
    RCS = 0.1;
    object = 'var';
    SNR = 20; 
    m = 1;
    shift_x = 0.05; 
    shift_y = -0.04;

    switch(f) 
        case 'f1'
            Freq = linspace(1.7,4.2,21)*1e9;
        case 'f3'
            Freq =(1.7:0.05:2.2)*1e9;               
        case 'f5'
            Freq = [1.79 2.03 2.2]*1e9; 
        case 'f6'
            Freq = 2e9;
        case 'f7'
            Freq = [1.8 2]*1e9;
        otherwise
            display('Error in frequency');
    end
    Nfreq = length(Freq);
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
        case 16
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

        otherwise
            disp('Error in Pos_recv');
    end
    figure;
    plot(Pos_tag(:,1),Pos_tag(:,2),'ro');
    hold on;
    plot(Pos_recv(:,1),Pos_recv(:,2),'bo','MarkerFaceColor','b','MarkerSize',8);
    hold on
    switch(object)
    case 'var'
        
        Pos_s = [0.0, 0; 
                0.02, 0.25;
                -0.1, 0.06;
                0.18,0.12;
                -0.28,-0.25;
                0.3,-0.1;
                -0.08,-0.3;
                ]; 
    case 'box1'
        
        % Ns = 4
        Pos_s = [-0.05,-0.05;
                 0.05,-0.05;
                 0.05,0.05;
                 -0.05,0.05];
    case 'circlefill'
        
        Pos_s=[];
        i=0;
        dtheta=1.2*pi/8;
        
        a0 = 0.04; % radius = a0*m
        % The radius of the object is a*m = 20*.25 cm = 5 cm
        for a = 0.02:0.02:a0
            theta=0;
            while theta<=2*pi-dtheta/2
                i=i+1;
                % Parametric form is (X,Y)=a(cos t, sin t)
                Pos_s(i,:)=(a.*[cos(theta) sin(theta)]);
                theta=theta+dtheta;
            end
            i=i+1;
            dtheta=dtheta-pi/64;
        end
%         Pos_s=sd_round(Pos_s); 
        
        %Radius
        ro=a*m*100;
        assignin('base','ro',ro);
    case 'circleouter'
         Pos_s=[];
        i=0;
        dtheta=pi/8;
        % The radius of the object is a*m = 20*.25 cm = 5 cm
        a=0.2;
        theta=0;
        while theta<=2*pi-dtheta/2
            i=i+1;
            % Parametric form is (X,Y)=a(cos t, sin t)
            Pos_s(i,:)=(a.*[cos(theta) sin(theta)]);
            theta=theta+dtheta;
        end
                
        %Radius
        ro=a*m*100;
        assignin('base','ro',ro);
    case 'ellipsefill'
        Pos_s=[];
        i=0;
        dtheta=pi/8;
        
        % The semi-major and semi-minor axes are (a, b)*m
        % phi is the angle by which it is rotated
        % Parametric form is (X,Y)=(a cos t, b sin t)
        % Parametric form for rotated is (X,Y)= (a*cos t*cos phi-b*sin t
        % sin phi, a*cos t*sin phi+b*sin t*cos phi)
        
        phi=pi/3;
        b=0.018;
        
        for a = 0.01:0.02:0.06
            theta=0;
            while theta<=2*pi-dtheta/2
                i=i+1;
                % Parametric form is (X,Y)=a(cos t, sin t)
                Pos_s(i,:)=[a.*cos(theta).*cos(phi)-b.*sin(theta).*sin(phi), a.*cos(theta).*sin(phi)+b.*sin(theta).*cos(phi)];
                theta=theta+dtheta;
            end
            i=i+1;
%             dtheta=dtheta-pi/32;
        end
        ao=a*m*100;
        bo=b*m*100;
        assignin('base','ao',ao);
        assignin('base','bo',bo);
    case 'boxfill'
        Pos_s=[];
        i=1;
        for x=-0.2:0.05:0.2
            for y=-0.2:0.05:0.2
                Pos_s(i,:)=[x y];
                i=i+1;
            end
            i=i+1;
        end
    otherwise 
        disp('Error in Pos_s');
    end
    Pos_s = m * Pos_s;
    [m1,~]=size(Pos_s);
    % Shifting Pos_s by k
    Pos_s = Pos_s - shift_x* [zeros(m1,1), ones(m1,1)];
    Pos_s = Pos_s - shift_y* [ones(m1,1), zeros(m1,1)];
    % Plotting object
    plot(Pos_s(:,1),Pos_s(:,2),'k*');
    axis 'square'
    axis 'tight'
    legend('Pos tag','Pos recv','Pos Scatterer');
    
    % Generate Rician Fading Factors
    An=sqrt(1/2*10^(-SNR/10));
    [shift_x, ~]=size(Pos_tag);
    [shift_y, ~]=size(Pos_recv);
    Rician_Factor=zeros(shift_x,shift_y,length(Freq));
    Rician_Factor_Phase=zeros(length(Freq)*shift_x*shift_y,1);
    Rician_Factor_Amp=zeros(length(Freq)*shift_x*shift_y,1);
    count =1;
    for i=1:length(Freq)
        for m=1:shift_x
            for n=1:shift_y
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
    G_wo_s=generation_signal_wo_scatter(Freq,Pos_tag,Pos_recv,Rician_Factor);
    assignin('base','G_wo_s',G_wo_s);
    G_w_s=generation_signal_w_scatter(Freq,Pos_tag,Pos_recv,Pos_s,RCS,Rician_Factor);
    assignin('base','G_w_s',G_w_s);

    % generate signal after calibration 
    [shift_x, ~]=size(Pos_tag);
    [shift_y, ~]=size(Pos_recv);
    G_calib=zeros(shift_x,shift_y,length(Freq));

    for i=1:length(Freq)
        freq=Freq(i);
        lamda=c/freq;
        for n=1:shift_y
            for m=1:shift_x
                p_tag=Pos_tag(m,:);
                p_recv=Pos_recv(n,:);
                r=norm(p_tag-p_recv);
                for n_pair=1:shift_y
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

    %% Generate e == A matrix, G == y (y = Ax)
    x_v = linspace(-0.4,0.4,101);
    y_v = x_v;
    lx = length(x_v);
    [X, Y] = meshgrid(x_v,y_v);
    P = combvec(x_v,y_v)';

    R1 = repmat(sqrt(sum(abs( repmat(permute(P, [1 3 2]), [1 Ntag 1]) ...
    - repmat(permute(Pos_tag, [3 1 2]), [lx*lx 1 1]) ).^2, 3)),1,Nrecv);
    dummy1 = ones(1,Ntag);
    R2 = kron(sqrt(sum(abs( repmat(permute(P, [1 3 2]), [1 Nrecv 1]) ...
    - repmat(permute(Pos_recv, [3 1 2]), [lx*lx 1 1]) ).^2, 3)), dummy1);
    R1 = repmat(R1,1,Nfreq); % All tag, recv, freq
    R2 = repmat(R2,1,Nfreq); % All tag, recv, freq
    dummy3 = ones(1,Ntag*Nrecv);
    Freq0 = repmat(kron(Freq,dummy3),[lx*lx 1]);
    e1 = exp(-1j*(2*pi/c)*(Freq0.*(R1+R2)));
    e_fwd = sqrt(RCS/(4*pi))*(1/(4*pi))*(c./(Freq0.*R1.*R2)).*e1; 
    e_fwd = transpose(e_fwd); % == A matrix
    e = e_fwd;
    V = reshape(G_calib,[],1); 
    % e_bwd check constant factor
    minfreq = min(Freq);
    e_bwd = transpose(((Freq0/minfreq).^2) .* e1); 
    B = e_bwd'*V; B = abs(B).^2;
    B = (reshape(B, [lx lx]));
    figure;
    mesh(X,Y,B')
    axis ([-0.5 0.5 -0.5 0.5])
    axis 'square'
    xlabel('x-axis','FontSize',14)
    ylabel('y-axis','FontSize',14)
    view(0,90)
    assignin('base','e',e);
    assignin('base','V',V);
end

%% generating signal without scatterer
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

%% generating signal with scatterer 
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
%             G(m,n,i) = 0;
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


