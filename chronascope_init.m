
function chronascope_init(restore_paths)

    if nargin < 1
        restore_paths = 0;
    end

    if restore_paths == 1
        disp("restoring default paths")
        restoredefaultpath
    end

    path_main = fileparts(mfilename('fullpath'));

    paths_to_add = ["/experiments";...
        "/experiments/chaotic_systems";
        "/utils";...
        "/correlation_dimension";...
        "/pinn_ts_rec"];

    % add paths
    for i = 1 : length(paths_to_add)
        path_new = path_main + paths_to_add(i);
        if ~contains(path, path_new)
            addpath(path_new);
            fprintf('new added path : %s\n', path_new);
        end
    end

end




