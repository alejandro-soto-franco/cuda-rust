/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scalar floating-point intrinsics compile for `sm_75` and the PTX assembles.
//!
//! Their catalog floor was `sm_80` although the instructions exist on Turing,
//! so a kernel using `mul_rn_f32`, `fma_rn_f32` or `ex2_approx_f32` was rejected
//! at `sm_75`. This builds one kernel from those three intrinsics, compiles it
//! for `sm_75` through the standalone compiler, and has `ptxas` assemble the
//! result for `sm_75`.

use cuda_oxide_codegen::__private::GENERATED_INTRINSIC_MARKER_ATTR;
use cuda_oxide_codegen::experimental::{CodegenModule, CompileOptions, Compiler, Target};

use dialect_mir::ops::{MirFuncOp, MirLoadOp, MirReturnOp, MirStoreOp};
use dialect_mir::types::MirPtrType;
use dialect_nvvm::ops::{
    ScalarArithmeticFormatAttr, ScalarArithmeticOp, ScalarArithmeticOperationAttr,
    ScalarArithmeticRoundingAttr, ScalarArithmeticSaturationAttr, ScalarArithmeticSubnormalAttr,
    ScalarMathFormatAttr, ScalarMathOp, ScalarMathOperationAttr, ScalarMathPrecisionAttr,
    ScalarMathSubnormalAttr,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{StringAttr, TypeAttr},
        op_interfaces::SymbolOpInterface,
        types::{FP32Type, FunctionType},
    },
    context::Context,
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    value::Value,
};

/// `scalar_kernel(a: *const f32, b: *const f32, out: *mut f32)` computing
/// `out[0] = ex2.approx(fma.rn(a[0], b[0], mul.rn(a[0], b[0])))`.
fn build_scalar_kernel(module: &mut CodegenModule) {
    module.edit(|ctx, module| {
        let module_region = module.get_operation().deref(ctx).get_region(0);
        let module_block = {
            let existing = module_region.deref(ctx).iter(ctx).next();
            existing.unwrap_or_else(|| {
                let block = BasicBlock::new(ctx, None, vec![]);
                block.insert_at_back(module_region, ctx);
                block
            })
        };

        let f32_ty = FP32Type::get(ctx);
        let in_ptr_ty = MirPtrType::get_global(ctx, f32_ty.into(), false);
        let out_ptr_ty = MirPtrType::get_global(ctx, f32_ty.into(), true);
        let func_type = FunctionType::get(
            ctx,
            vec![in_ptr_ty.into(), in_ptr_ty.into(), out_ptr_ty.into()],
            vec![],
        );
        let func_op = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let func = MirFuncOp::new(ctx, func_op, TypeAttr::new(func_type.into()));
        func.set_symbol_name(ctx, "scalar_kernel".try_into().unwrap());
        let entry = BasicBlock::new(
            ctx,
            None,
            vec![in_ptr_ty.into(), in_ptr_ty.into(), out_ptr_ty.into()],
        );
        entry.insert_at_front(func_op.deref(ctx).get_region(0), ctx);
        let (a, b, out) = {
            let block = entry.deref(ctx);
            (
                block.get_argument(0),
                block.get_argument(1),
                block.get_argument(2),
            )
        };

        // The generated catalog marker names the exact intrinsic variant, the
        // way the rustc importer tags each call.
        let tag = |ctx: &mut Context, op: pliron::context::Ptr<Operation>, marker: &str| {
            op.deref_mut(ctx).attributes.set(
                Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                StringAttr::new(marker.to_owned()),
            );
        };
        let emit = |ctx: &mut Context, op: pliron::context::Ptr<Operation>| -> Option<Value> {
            op.insert_at_back(entry, ctx);
            (op.deref(ctx).get_num_results() > 0).then(|| op.deref(ctx).get_result(0))
        };
        let load = |ctx: &mut Context, pointer: Value| {
            let op = Operation::new(
                ctx,
                MirLoadOp::get_concrete_op_info(),
                vec![f32_ty.into()],
                vec![pointer],
                vec![],
                0,
            );
            emit(ctx, op).unwrap()
        };
        let a_val = load(ctx, a);
        let b_val = load(ctx, b);

        let arithmetic = |ctx: &mut Context, operands: Vec<Value>, operation, marker: &str| {
            let op = ScalarArithmeticOp::build(
                ctx,
                operands,
                ScalarArithmeticFormatAttr::F32,
                operation,
                ScalarArithmeticRoundingAttr::Rn,
                ScalarArithmeticSubnormalAttr::Preserve,
                ScalarArithmeticSaturationAttr::None,
            );
            tag(ctx, op, marker);
            emit(ctx, op).unwrap()
        };
        let product = arithmetic(
            ctx,
            vec![a_val, b_val],
            ScalarArithmeticOperationAttr::Mul,
            "v1:i0402",
        );
        let fused = arithmetic(
            ctx,
            vec![a_val, b_val, product],
            ScalarArithmeticOperationAttr::Fma,
            "v1:i0418",
        );
        let exp2 = {
            let op = ScalarMathOp::build(
                ctx,
                fused,
                ScalarMathFormatAttr::F32,
                ScalarMathOperationAttr::Ex2,
                ScalarMathPrecisionAttr::Approx,
                ScalarMathSubnormalAttr::Preserve,
            );
            tag(ctx, op, "v1:i0819");
            emit(ctx, op).unwrap()
        };

        let store = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![out, exp2],
            vec![],
            0,
        );
        emit(ctx, store);
        let ret = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        emit(ctx, ret);

        func.get_operation().insert_at_back(module_block, ctx);
    });
    module.mark_kernel_entry("scalar_kernel").unwrap();
}

/// Locate `ptxas`: `CUDA_TOOLKIT_PATH`, then `CUDA_HOME`, then
/// `/usr/local/cuda`, then `PATH`.
fn find_ptxas() -> std::path::PathBuf {
    ["CUDA_TOOLKIT_PATH", "CUDA_HOME"]
        .iter()
        .filter_map(|var| std::env::var(var).ok())
        .filter(|root| !root.trim().is_empty())
        .map(|root| std::path::PathBuf::from(root).join("bin/ptxas"))
        .chain(std::iter::once(std::path::PathBuf::from(
            "/usr/local/cuda/bin/ptxas",
        )))
        .find(|candidate| candidate.exists())
        .unwrap_or_else(|| std::path::PathBuf::from("ptxas"))
}

#[test]
fn scalar_float_intrinsics_compile_and_assemble_for_sm75() {
    let mut module = CodegenModule::new("scalar_sm75").unwrap();
    build_scalar_kernel(&mut module);
    let compiler = Compiler::discover().expect("LLVM 21+ llc/opt are installed");
    let ptx = compiler
        .compile(
            &mut module,
            &CompileOptions::new(Target::parse("sm_75").unwrap()),
        )
        .expect("scalar intrinsics compile for sm_75")
        .into_ptx();
    let text = String::from_utf8(ptx.clone()).expect("PTX is utf-8");

    assert!(text.contains(".target sm_75"), "PTX targets sm_75:\n{text}");
    for instruction in ["mul.rn.f32", "fma.rn.f32", "ex2.approx.f32"] {
        assert!(
            text.contains(instruction),
            "{instruction} survives:\n{text}"
        );
    }

    let dir = std::env::temp_dir().join(format!(
        "scalar_float_sm75_ptx_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let ptx_path = dir.join("scalar.ptx");
    std::fs::write(&ptx_path, &ptx).unwrap();
    let ptxas = find_ptxas();
    let result = std::process::Command::new(&ptxas)
        .arg("-arch=sm_75")
        .arg("--compile-only")
        .arg(&ptx_path)
        .arg("-o")
        .arg(dir.join("scalar.cubin"))
        .output();
    let _ = std::fs::remove_dir_all(&dir);
    let out = result.unwrap_or_else(|error| {
        panic!(
            "could not run {}: {error}\nThis test needs `ptxas` from a CUDA toolkit (no GPU \
             required). Point CUDA_TOOLKIT_PATH or CUDA_HOME at the install root, or put \
             `ptxas` on PATH.",
            ptxas.display()
        )
    });
    assert!(
        out.status.success(),
        "ptxas rejected the sm_75 PTX:\n{}\n\nPTX:\n{text}",
        String::from_utf8_lossy(&out.stderr)
    );
}
