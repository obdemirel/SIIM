% compute D'_x(U)
function dxtu = DxtU(U)
    dxtu = cat(2, U(:,end,:) - U(:,1,:), -diff(U,1,2));
%    [U(:,end)-U(:, 1) U(:,1:end-1)-U(:,2:end)];
end
