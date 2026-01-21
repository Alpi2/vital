# Additional backup and restore endpoints to add to demo.py

# Add these endpoints after the existing reset endpoint in demo.py:

@router.post("/backups", 
         response_model=BackupResponse,
         summary="Create demo data backup",
         description="Create a backup of current demo data with selective options")
@limiter.limit("10/minute")
async def create_backup(
    request: BackupRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(rbac_service.get_current_user)
) -> Dict[str, Any]:
    """Create a backup of current demo data"""
    try:
        user_id = current_user.get("sub") if current_user else None
        user_name = current_user.get("name", "Unknown User") if current_user else "System"
        
        result = await demo_data_manager.create_backup(
            name=request.name,
            description=request.description,
            backup_type=request.backup_type,
            include_patients=request.include_patients,
            include_ecg_data=request.include_ecg_data,
            include_alerts=request.include_alerts,
            include_devices=request.include_devices,
            user_id=user_id,
            user_name=user_name
        )
        
        return BackupResponse(
            success=result['success'],
            backup_id=result.get('backup_id'),
            name=result.get('name', 'Unknown'),
            size_bytes=result.get('size_bytes', 0),
            created_at=result.get('created_at', datetime.utcnow().isoformat()),
            duration_ms=result.get('duration_ms', 0),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"Backup creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backups", 
        response_model=List[Dict[str, Any]],
        summary="List available backups",
        description="Get list of all available demo data backups")
@limiter.limit("50/minute")
async def list_backups(
    limit: int = Query(50, description="Maximum number of backups to return"),
    current_user = Depends(rbac_service.get_current_user)
) -> List[Dict[str, Any]]:
    """List all available backups"""
    try:
        user_id = current_user.get("sub") if current_user else None
        return await demo_data_manager.get_backups(user_id=user_id, limit=limit)
        
    except Exception as e:
        logger.error(f"Failed to list backups: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backups/{backup_id}/restore", 
         response_model=RestoreResponse,
         summary="Restore from backup",
         description="Restore demo data from a specific backup")
@limiter.limit("5/minute")
async def restore_from_backup(
    backup_id: str,
    request: RestoreRequest,
    current_user = Depends(rbac_service.get_current_user)
) -> Dict[str, Any]:
    """Restore demo data from backup"""
    if not request.confirm:
        raise HTTPException(status_code=400, detail="Confirmation required for restore operation")
    
    try:
        user_id = current_user.get("sub") if current_user else None
        user_name = current_user.get("name", "Unknown User") if current_user else "System"
        
        result = await demo_data_manager.restore_from_backup(
            backup_id=backup_id,
            user_id=user_id,
            user_name=user_name
        )
        
        return RestoreResponse(
            success=result['success'],
            backup_id=result['backup_id'],
            backup_name=result.get('backup_name', 'Unknown'),
            restored_at=result.get('restored_at', datetime.utcnow().isoformat()),
            duration_ms=result.get('duration_ms', 0),
            records_restored=result.get('records_restored', 0),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"Backup restore failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/backups/{backup_id}", 
           summary="Delete backup",
           description="Delete a specific demo data backup")
@limiter.limit("10/minute")
async def delete_backup(
    backup_id: str,
    current_user = Depends(rbac_service.get_current_user)
) -> Dict[str, Any]:
    """Delete a backup"""
    try:
        user_id = current_user.get("sub") if current_user else None
        result = await demo_data_manager.delete_backup(backup_id=backup_id, user_id=user_id)
        
        if result['success']:
            return {
                "success": True,
                "backup_id": result['backup_id'],
                "deleted_at": result['deleted_at']
            }
        else:
            raise HTTPException(status_code=404, detail=result['error'])
            
    except Exception as e:
        logger.error(f"Backup deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reset-history", 
        response_model=ResetHistoryResponse,
        summary="Get reset history",
        description="Get history of all reset operations")
@limiter.limit("50/minute")
async def get_reset_history(
    limit: int = Query(50, description="Maximum number of history entries to return"),
    current_user = Depends(rbac_service.get_current_user)
) -> Dict[str, Any]:
    """Get reset operation history"""
    try:
        user_id = current_user.get("sub") if current_user else None
        history = await demo_data_manager.get_reset_history(user_id=user_id, limit=limit)
        
        return ResetHistoryResponse(
            history=history,
            total_count=len(history)
        )
        
    except Exception as e:
        logger.error(f"Failed to get reset history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
