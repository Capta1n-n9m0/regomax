// noinspection DuplicatedCode

const fsp = require("fs").promises;
const path = require("path");
const bsplit = require("buffer-split");
const Network = require("./Network");
const Matrix = require("./Matrix");
const printf = require("printf");
const Vector = require("./Vector");
const os = require("os");
const nCpu = os.cpus().length;
const { Worker } = require("worker_threads");

const eps_pagerank = 1e-13;

/**
 * @return {number}
 */
function getTime() {
    return Number(process.hrtime.bigint() / BigInt(1_000_000));
}

let calc_pg_proj_th_id = 0;
/**
 * @param{Vector} pagerank
 * @param{Network} net
 * @param{number} delta_alpha
 * @param{number} iprint
 * @param{Vector} node
 * @param{number} trans_frag
 * @return {Promise<number>}
 */
async function calc_pg_proj_th(pagerank, net, delta_alpha, iprint, node, trans_frag = 0){
    return new Promise((resolve)=>{
        let w = new Worker(path.join(__dirname, "Thread.js"));
        w.on("message", (msg)=>{
            resolve(msg.data);
        });
        w.postMessage({
            options:{
                work: true,
                task: 1,
                once: true,
                id: calc_pg_proj_th_id
            },
            data: {pagerank, net, delta_alpha, iprint, node, trans_frag}
        });
        calc_pg_proj_th_id++;
    });
}

/**
 * @param{Vector} right
 * @param{Vector} left
 * @param{Vector} pg
 * @param{Network} net
 * @param{number} delta_alpha
 * @param{Vector} node
 * @return Promise<number>
 */
async function compute_project(right, left, pg, net, delta_alpha, node) {
    let f_timer = getTime();
    const iprint = 10;
    let sp, dlambda1, dlambda2, dlambda3;
    let node0 = new Vector(0);

    right.put_value(1.0);
    left.put_value(1.0);
    pg.put_value(1.0);

    let p2 = calc_pg_proj_th(left, net, delta_alpha, iprint, node, 1);
    let p1 = calc_pg_proj_th(right, net, delta_alpha, iprint, node);
    let p3 = calc_pg_proj_th(pg, net, delta_alpha, iprint, node0);

    dlambda1 = await p1;
    dlambda2 = await p2;
    dlambda3 = await p3;

    sp = 1.0 / Vector.scalar_product(left, right);
    left.mul_eq(sp);

    // sp = Vector.scalar_product(left, right);
    // console.log(printf("dlambda = %24.16f   diff = %f\n",
    //     dlambda1, Math.abs(dlambda1 - dlambda2)));
    // console.log(printf("TEST: psi_left^T * psi_right = %26.16f\n", sp));

    console.log(`compute_project : ${getTime() - f_timer} ms`);
    return dlambda1;
}

/**
 * @type {Worker[]}
 */
let threads = new Array(nCpu);

async function send_data(data){
    return new Promise((resolve)=>{
        let done_counter = 0;
        for(let i = 0; i < threads.length; i++){
            threads[i].executor = (msg)=>{
                done_counter++;
                if(done_counter === threads.length){
                    for(let j = 0; j < threads[j]; j++){
                        threads[j].off("message", threads[j].executor);
                    }
                    resolve(1);
                }
            }
            threads[i].on("message", threads[i].executor);
        }
        for (let i = 0; i < threads.length; i++) {
            threads[i].postMessage({data, options:{work: true, once: false, stage: 1, id: i, task: 2}})
        }
    })
}

async function compute_GR_heavy(data){
    return new Promise(async (resolve) => {
        let nr = data.node.dim;
        for (let i = 0; i < nCpu; i++) {
            threads[i] = new Worker(path.join(__dirname, "Thread.js"));
        }
        await send_data(data);
        let c = -1;
        let done_counter = 0;
        for(let i = 0; i < threads.length; i++){
            threads[i].executor = (msg)=>{
                c++;
                done_counter++;
                if(c < nr){
                    threads[i].postMessage({data:{i: c}, options:{work:true, once: false, stage: 2, task: 2}})
                }
                if(done_counter === nr){
                    for(let j = 0; j < threads.length; j++){
                        threads[j].postMessage({options: {work: false}});
                        threads[j].off("message", threads[j].executor);
                    }
                    resolve();
                }
            }
            threads[i].on("message", threads[i].executor);
        }
        for(let i = 0; i < threads.length; i++){
            c++;
            if(c < nr){
                threads[i].postMessage({data:{i: c}, options:{work: true, once: false, stage: 2, task: 2}});
            }
        }
    });
}

/**
 * @param{Matrix} G_R
 * @param{Matrix} G_rr
 * @param{Matrix} G_pr
 * @param{Matrix} G_qr
 * @param{Matrix} G_I
 * @param{Vector} psiL
 * @param{Vector} psiR
 * @param{Vector} pg
 * @param{Network} net
 * @param{number} delta_alpha
 * @param{Vector} node
 */
async function compute_GR(G_R, G_rr, G_pr,
                          G_qr, G_I, psiL,
                          psiR, pg, net,
                          delta_alpha, node) {
    let nr = node.dim;
    if (G_R.x !== nr || G_R.y !== nr) throw "Wrong matrix size of G_R  in compute_GR";
    if (G_rr.x !== nr || G_rr.y !== nr) throw "Wrong matrix size of G_rr  in compute_GR";
    if (G_pr.x !== nr || G_pr.y !== nr) throw "Wrong matrix size of G_pr  in compute_GR";
    if (G_qr.x !== nr || G_qr.y !== nr) throw "Wrong matrix size of G_qr  in compute_GR";
    if (G_I.x !== nr || G_I.y !== nr) throw "Wrong matrix size of G_I  in compute_GR";
    let dlambda;

    let max_iter;

    max_iter = Math.floor(-Math.log(eps_pagerank) / (delta_alpha + 3e-7));
    max_iter *= 2;

    console.log("Computation of left and right eigenvectors of G_ss");
    dlambda = await compute_project(psiR, psiL, pg, net, delta_alpha, node);

    let c_GR_h_timer = getTime();
    await compute_GR_heavy({G_R, G_rr, G_pr, G_qr, G_I, psiL, psiR, pg, net, delta_alpha, node, max_iter, dlambda});
    console.log(`compute_GR loop : ${getTime() - c_GR_h_timer} ms`);
}

async function main(argv) {
    const start = getTime();
    const net_file = argv[2];
    const delta_alpha = parseFloat(argv[3]);
    const iprint = parseInt(argv[4]);
    const print_number = parseInt(argv[5]);
    const ten_number = parseInt(argv[6]);
    let node_file = argv[7];
    const nodenames_file = argv[8];
    const limit = parseInt(argv[9]);
    console.log(`input-file     = ${path.resolve(net_file)}`);
    console.log(`1-alpha        = ${delta_alpha}`);
    console.log(`iprint         = ${iprint}`);
    console.log(`print_number   = ${print_number}`);
    console.log(`ten_number     = ${ten_number}`);
    console.log(`node_file      = ${path.resolve(node_file)}`);
    console.log(`nodenames_file = ${path.resolve(nodenames_file)}`);


    let nodes_read_timer = getTime();
    let tmp = [];
    const len = await fsp.readFile(node_file)
        .then(data => {
            const lines = bsplit(data, Buffer.from("\n"));
            for (const line of lines.slice(1)) {
                if (line.length === 0) continue;
                const n = parseInt(line.toString());
                if(limit) {
                    if (n <= limit) tmp.push(n);
                }
                else tmp.push(n);
            }
            return tmp.length;
        });
    let node = new Vector(len);
    for(let i = 0; i < tmp.length; i++){
        node.c[i] = tmp[i];
    }
    console.log(`Read node_file : ${len} nodes`);
    console.log(`Read node_file : ${getTime() - nodes_read_timer} ms`);
    const net = new Network(net_file, limit);
    let GR = new Matrix(len, len),
        Grr = new Matrix(len, len),
        Gpr = new Matrix(len, len),
        Gqr = new Matrix(len, len),
        GI = new Matrix(len, len);
    const n = net.size;
    let psiL = new Vector(n),
        psiR = new Vector(n),
        pg = new Vector(n);
    node_file = path.basename(node_file, ".nodes");
    let calc_timer = getTime();
    await compute_GR(GR, Grr, Gpr, Gqr, GI, psiL, psiR, pg, net, delta_alpha, node);
    console.log(`Calculations : ${getTime() - calc_timer} ms`);
    let write_timer = getTime();
    Matrix.print_mat(Gqr, `Gqr_${net.base_name}_${node_file}_${len}.dat`, nodenames_file);
    console.log(`Writing matrix : ${getTime() - write_timer} ms`);
    console.log(`Execution : ${getTime() - start} ms\n`);
    return 0;
}

module.exports = main;
